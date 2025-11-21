// ===== CONFIGURATION VARIABLES =====
const CONFIG = {
  // Number of words to display on screen at a time
  wordsToShow: 8,
  minWordsToShow: 3,
  
  // General subtitle styling
  fontName: 'Arial',          // Font face
  fontSize: 130,               // Base font size
  fontColor: 'FFFFFF',        // Regular text color in BGR format
  outlineColor: '000000',     // Text outline color in BGR format
  backgroundColor: '000000',  // Background color in BGR format
  
  // Highlight styling
  highlightFontSize: 130,      // Highlighted word font size
  highlightColor: '00FFFF',   // Highlight color in BGR format (cyan)
  
  // Text appearance
  bold: 0,                    // 0 = off, 1 = on
  italic: 0,                  // 0 = off, 1 = on
  underline: 0,               // 0 = off, 1 = on
  strikeout: 0,               // 0 = off, 1 = on
  
  // Text scaling and positioning
  scaleX: 100,                // Horizontal scaling (%)
  scaleY: 100,                // Vertical scaling (%)
  spacing: 0,                 // Letter spacing
  angle: 0,                   // Rotation angle
  
  // Border options
  borderStyle: 1,             // 1 = outline+drop shadow, 3 = opaque box
  outline: 10,                 // Outline thickness
  shadow: 1,                  // Shadow distance
  
  // Alignment and margins
  alignment: 2,               // Position (1-9), see ASS documentation
  marginL: 10,                // Left margin
  marginR: 10,                // Right margin
  marginV: 300,               // Vertical margin
  
  // Video dimensions
  videoWidth: 1920,
  videoHeight: 1080,
  
  
  // Maximum gap time between words (in milliseconds) before considering it a new phrase
  maxGapTimeMs: 2000,

  
  // Output file name
  outputFileName: 'sentence_aware_subtitles.ass'
};

// Parse input JSON data
const response = JSON.parse($json.data);
const segments = response.segments;

// ASS Header
let ass = `[Script Info]
Title: Sentence-Aware Fixed Position Subtitles
ScriptType: v4.00+
PlayResX: ${CONFIG.videoWidth}
PlayResY: ${CONFIG.videoHeight}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,${CONFIG.fontName},${CONFIG.fontSize},&H${CONFIG.fontColor},&H${CONFIG.outlineColor},&H${CONFIG.outlineColor},&H${CONFIG.backgroundColor},${CONFIG.bold},${CONFIG.italic},${CONFIG.underline},${CONFIG.strikeout},${CONFIG.scaleX},${CONFIG.scaleY},${CONFIG.spacing},${CONFIG.angle},${CONFIG.borderStyle},${CONFIG.outline},${CONFIG.shadow},${CONFIG.alignment},${CONFIG.marginL},${CONFIG.marginR},${CONFIG.marginV},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
`;

/**
 * Helper: convert seconds to ASS time format (H:MM:SS.cs)
 * @param {number} sec - Time in seconds
 * @return {string} Formatted time string
 */
function formatTime(sec) {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  const cs = Math.floor((sec - Math.floor(sec)) * 100);
  return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${cs.toString().padStart(2, '0')}`;
}

/**
 * Creates the subtitle text with the specified word highlighted within a fixed group
 * @param {Array} wordGroup - Array of word objects to display in fixed positions
 * @param {number} highlightIndex - Index of the word to highlight within the group (0-based)
 * @return {string} Formatted subtitle text with highlight tags
 */
function createSubtitleText(wordGroup, highlightIndex) {
  let line = '';
  
  wordGroup.forEach((word, i) => {
    // Add space between words (not before the first word), unless the word starts with '-'
    if (i > 0 && !word.word.trim().startsWith('-')) line += ' ';
    
    // Apply highlight to the specified word
    if (i === highlightIndex) {
      // Mark the highlighted word with a special tag for post-processing
      // We keep the font size change here so the layout is roughly correct, 
      // but the Python script will handle the final positioning and animation.
      line += `{\\fs${CONFIG.highlightFontSize}\\c&H${CONFIG.highlightColor}&\\highlight}${word.word.trim()}{\\r}`;
    } else {
      line += word.word.trim();
    }
  });
  
  return line;
}

// Process all segments together to handle pauses between segments
let allWords = [];
segments.forEach(segment => {
  allWords = allWords.concat(segment.words);
});

// Sort all words by start time (should already be sorted, but just to be safe)
allWords.sort((a, b) => a.start - b.start);

// Merge '%' with previous word
for (let i = 1; i < allWords.length; i++) {
  const currentWord = allWords[i];
  const prevWord = allWords[i - 1];
  
  if (currentWord.word.trim() === '%') {
    prevWord.word = prevWord.word.trim() + '%';
    prevWord.end = currentWord.end;
    allWords.splice(i, 1);
    i--; // Adjust index since we removed an element
  }
}

// Create groups of words that respect sentence boundaries
function createSentenceAwareGroups(words) {
  const sentences = [];
  let currentSentence = [];
  
  // Step 1: Group into sentences
  for (let i = 0; i < words.length; i++) {
    currentSentence.push(words[i]);
    const w = words[i].word.trim();
    // Check for sentence terminators
    if (w.endsWith('.') || w.endsWith('!') || w.endsWith('?') || i === words.length - 1) {
       sentences.push(currentSentence);
       currentSentence = [];
    }
  }
  
  const groups = [];
  
  // Step 2: Process each sentence
  for (const sentence of sentences) {
     let remainingWords = [...sentence];
     
     while (remainingWords.length > 0) {
        let count = Math.min(remainingWords.length, CONFIG.wordsToShow);
        const remainingAfter = remainingWords.length - count;
        
        // If the remainder would be less than minWordsToShow (and not zero),
        // reduce the current count to leave enough for the next group.
        if (remainingAfter > 0 && remainingAfter < CONFIG.minWordsToShow) {
           count = remainingWords.length - CONFIG.minWordsToShow;
           // Ensure we don't end up with 0 or negative count
           if (count < 1) count = 1; 
        }
        
        groups.push(remainingWords.slice(0, count));
        remainingWords = remainingWords.slice(count);
     }
  }
  
  return groups;
}

// Create word groups that respect sentence boundaries
const wordGroups = createSentenceAwareGroups(allWords);
let dialogueLines = [];

// Track which group and position each word belongs to
let currentGroupIndex = 0;
let positionInGroup = 0;

for (let i = 0; i < allWords.length; i++) {
  const currentWord = allWords[i];
  
  // Find which group contains this word
  let groupIndex = -1;
  let positionIndex = -1;
  
  for (let g = 0; g < wordGroups.length; g++) {
    const wordIndex = wordGroups[g].findIndex(w => 
      w.start === currentWord.start && w.end === currentWord.end && w.word === currentWord.word);
    
    if (wordIndex !== -1) {
      groupIndex = g;
      positionIndex = wordIndex;
      break;
    }
  }
  
  if (groupIndex === -1) continue; // Skip if not found (should never happen)
  
  const wordGroup = wordGroups[groupIndex];
  
  // Create subtitle with the correct word highlighted
  const startTime = currentWord.start;
  const endTime = currentWord.end;
  const formattedStart = formatTime(startTime);
  const formattedEnd = formatTime(endTime);
  const subtitleText = createSubtitleText(wordGroup, positionIndex);
  
  // Add this subtitle to our list
  dialogueLines.push({
    start: startTime,
    end: endTime,
    formattedStart,
    formattedEnd,
    text: subtitleText,
    group: groupIndex,
    position: positionIndex
  });
}

// Now, fill in any gaps between dialogues of the same word group
for (let i = 0; i < dialogueLines.length - 1; i++) {
  const current = dialogueLines[i];
  const next = dialogueLines[i + 1];
  
  // If they're in the same group and there's a gap
  if (current.group === next.group && next.start - current.end > 0 && next.start - current.end <= CONFIG.maxGapTimeMs / 1000) {
    // Fill the gap by extending the current line's end time
    current.end = next.start;
    current.formattedEnd = formatTime(current.end);
  }
}

// Write all dialogue lines to the ASS file
dialogueLines.forEach(line => {
  ass += `Dialogue: 0,${line.formattedStart},${line.formattedEnd},Default,,0,0,0,,${line.text}\n`;
});

// Return as binary file
return [{
  json: {},
  binary: {
    data: {
      data: Buffer.from(ass).toString('base64'),
      mimeType: 'text/ass',
      fileName: CONFIG.outputFileName
    }
  }
}];

