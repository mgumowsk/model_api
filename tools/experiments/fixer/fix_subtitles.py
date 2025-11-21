import json
import re
import difflib
import sys

def normalize(s):
    # Keep alphanumeric and %
    return re.sub(r'[^a-z0-9%]', '', s.lower())

def fix_json(out_path, org_path):
    print(f"Reading {out_path} and {org_path}...")
    with open(out_path, 'r') as f:
        out_data = json.load(f)
    
    with open(org_path, 'r') as f:
        org_text = f.read()

    # Flatten out words
    out_words = []
    for seg in out_data['segments']:
        if 'words' in seg:
            out_words.extend(seg['words'])

    # Tokenize org text
    raw_tokens = org_text.split()
    org_tokens = []
    for token in raw_tokens:
        if '-' in token:
            parts = re.split(r'(?=-)', token)
            org_tokens.extend([p for p in parts if p])
        else:
            org_tokens.append(token)

    # Build alignment strings
    out_align_str = ""
    out_map = [] # char_index -> word_index
    
    for i, w in enumerate(out_words):
        norm = normalize(w['word'])
        # If norm is empty (e.g. just punctuation), we might skip it for alignment
        # but we should keep track of it? 
        # For now, let's skip empty norms in alignment source.
        if norm:
            start_idx = len(out_align_str)
            out_align_str += norm
            end_idx = len(out_align_str)
            out_map.extend([i] * (end_idx - start_idx))

    org_align_str = ""
    org_token_spans = [] # index -> (start, end) in org_align_str
    
    for i, w in enumerate(org_tokens):
        norm = normalize(w)
        start_idx = len(org_align_str)
        if norm:
            org_align_str += norm
        end_idx = len(org_align_str)
        org_token_spans.append((start_idx, end_idx))

    print("Aligning sequences...")
    sm = difflib.SequenceMatcher(None, out_align_str, org_align_str)
    opcodes = sm.get_opcodes()

    new_words = []
    
    # Helper to find out_indices for a range in org_align_str
    def get_source_indices(org_start, org_end):
        indices = set()
        for tag, i1, i2, j1, j2 in opcodes:
            # Intersection of [j1, j2) (org range in opcode) and [org_start, org_end)
            start_overlap = max(j1, org_start)
            end_overlap = min(j2, org_end)
            
            if start_overlap < end_overlap:
                if tag == 'equal':
                    # Map to i1 + offset
                    out_start_offset = start_overlap - j1
                    out_end_offset = end_overlap - j1
                    for k in range(i1 + out_start_offset, i1 + out_end_offset):
                        if k < len(out_map):
                            indices.add(out_map[k])
                elif tag == 'replace':
                    # Map to the whole replaced range in out
                    for k in range(i1, i2):
                        if k < len(out_map):
                            indices.add(out_map[k])
                # insert: no source
                # delete: ignored
        return indices

    print("Processing tokens...")
    # First pass: determine sources for each token
    token_sources = []
    for i, token in enumerate(org_tokens):
        start, end = org_token_spans[i]
        if start == end:
            # Empty norm (punctuation only)
            # Try to attach to previous or next?
            # For now, empty sources
            token_sources.append(set())
        else:
            sources = get_source_indices(start, end)
            token_sources.append(sources)

    # Second pass: resolve shared sources and calculate timings
    final_tokens = []
    
    for i, token in enumerate(org_tokens):
        sources = token_sources[i]
        
        # If no sources, we need interpolation
        if not sources:
            final_tokens.append({
                'word': token,
                'start': None,
                'end': None,
                'interpolated': True
            })
            continue

        # Check if these sources are shared with other tokens
        # We group consecutive tokens that share the EXACT same set of sources
        # (This is a simplification, but handles the "7%." -> "seven", "%" case)
        
        # Actually, we can just calculate the full range of the sources
        # and then split it if multiple tokens map to it.
        
        source_list = sorted(list(sources))
        min_start = min(out_words[idx]['start'] for idx in source_list)
        max_end = max(out_words[idx]['end'] for idx in source_list)
        
        final_tokens.append({
            'word': token,
            'start': min_start,
            'end': max_end,
            'interpolated': False,
            'source_indices': sources
        })

    # Fix shared timings (split them)
    i = 0
    while i < len(final_tokens):
        if final_tokens[i]['interpolated']:
            i += 1
            continue
            
        # Find group of tokens with same start/end (implying shared sources or just coincidence)
        # Better: check source_indices equality?
        # But "2" and "%" -> "2%" (one token) has sources {2, %}.
        # "7%." -> "seven", "%" (two tokens).
        # "seven" has sources {7%.}. "%" has sources {7%.}.
        # They will have same start/end.
        
        j = i + 1
        group = [i]
        while j < len(final_tokens):
            if final_tokens[j]['interpolated']:
                break
            if (final_tokens[j]['start'] == final_tokens[i]['start'] and 
                final_tokens[j]['end'] == final_tokens[i]['end']):
                group.append(j)
                j += 1
            else:
                break
        
        if len(group) > 1:
            # Split the duration
            total_duration = final_tokens[i]['end'] - final_tokens[i]['start']
            # Split by length of word? Or equal?
            # "seven" (5) vs "%" (1).
            total_len = sum(len(normalize(final_tokens[k]['word'])) for k in group)
            if total_len == 0: total_len = len(group) # Fallback
            
            current_time = final_tokens[i]['start']
            for k in group:
                w_len = len(normalize(final_tokens[k]['word']))
                if w_len == 0: w_len = 1 # Give punctuation some weight
                
                duration = (w_len / total_len) * total_duration if total_len > 0 else total_duration / len(group)
                
                final_tokens[k]['start'] = current_time
                final_tokens[k]['end'] = current_time + duration
                current_time += duration
                
        i = j

    # Interpolate missing timings
    for i in range(len(final_tokens)):
        if final_tokens[i]['interpolated']:
            # Find prev end
            prev_end = 0.0
            if i > 0 and final_tokens[i-1]['end'] is not None:
                prev_end = final_tokens[i-1]['end']
            
            # Find next start
            next_start = prev_end + 0.1 # Default duration if at end
            for j in range(i + 1, len(final_tokens)):
                if final_tokens[j]['start'] is not None:
                    next_start = final_tokens[j]['start']
                    break
            
            final_tokens[i]['start'] = prev_end
            final_tokens[i]['end'] = next_start
            
            # If we have a sequence of interpolated tokens, we should distribute the gap
            # But the loop handles one by one. 
            # If i+1 is also interpolated, next_start will be found at i+2...
            # Wait, the inner loop finds the first NON-interpolated token.
            # So next_start is correct for the whole gap.
            # We need to distribute (next_start - prev_end) among the gap tokens.
            
            # Let's do a proper gap fill pass
            pass

    # Proper gap fill
    i = 0
    while i < len(final_tokens):
        if final_tokens[i]['interpolated']:
            # Start of a gap
            gap_start_idx = i
            # Find end of gap
            while i < len(final_tokens) and final_tokens[i]['interpolated']:
                i += 1
            gap_end_idx = i
            
            # Gap is from gap_start_idx to gap_end_idx (exclusive)
            
            prev_time = 0.0
            if gap_start_idx > 0:
                prev_time = final_tokens[gap_start_idx-1]['end']
            
            next_time = prev_time + 0.5 # Default
            if gap_end_idx < len(final_tokens):
                next_time = final_tokens[gap_end_idx]['start']
            
            duration = next_time - prev_time
            if duration < 0: duration = 0
            
            gap_len = gap_end_idx - gap_start_idx
            step = duration / gap_len
            
            for k in range(gap_start_idx, gap_end_idx):
                final_tokens[k]['start'] = prev_time + (k - gap_start_idx) * step
                final_tokens[k]['end'] = prev_time + (k - gap_start_idx + 1) * step
                final_tokens[k]['interpolated'] = False
        else:
            i += 1

    # Construct output
    # We'll put everything in one segment
    new_segments = [{
        "id": 0,
        "seek": 0,
        "start": final_tokens[0]['start'],
        "end": final_tokens[-1]['end'],
        "text": org_text,
        "words": final_tokens
    }]
    
    # Clean up final tokens (remove internal keys)
    for t in final_tokens:
        if 'interpolated' in t: del t['interpolated']
        if 'source_indices' in t: del t['source_indices']
        # Round timings
        t['start'] = round(t['start'], 2)
        t['end'] = round(t['end'], 2)
        # Ensure probability exists (dummy)
        t['probability'] = 1.0

    out_data['segments'] = new_segments
    out_data['text'] = org_text
    
    print(f"Writing to {out_path}...")
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=4)
    print("Done.")

if __name__ == "__main__":
    fix_json('out.json', 'org.txt')
