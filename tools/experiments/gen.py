import re
import ctypes
import os
import sys
import argparse

LIBASS_PATH = "/opt/homebrew/lib/libass.dylib"

class ASS_Image(ctypes.Structure):
    pass

ASS_Image._fields_ = [
    ("w", ctypes.c_int),
    ("h", ctypes.c_int),
    ("stride", ctypes.c_int),
    ("bitmap", ctypes.POINTER(ctypes.c_ubyte)),
    ("color", ctypes.c_uint32),
    ("dst_x", ctypes.c_int),
    ("dst_y", ctypes.c_int),
    ("next", ctypes.POINTER(ASS_Image)),
    ("type", ctypes.c_int)
]

class LibassMeasurer:
    def __init__(self, width=1920, height=1080):
        try:
            self.lib = ctypes.CDLL(LIBASS_PATH)
        except OSError:
            print(f"Could not load libass at {LIBASS_PATH}")
            raise

        # Define Function Signatures
        self.lib.ass_library_init.restype = ctypes.c_void_p
        self.lib.ass_library_init.argtypes = []

        self.lib.ass_renderer_init.restype = ctypes.c_void_p
        self.lib.ass_renderer_init.argtypes = [ctypes.c_void_p]

        self.lib.ass_new_track.restype = ctypes.c_void_p
        self.lib.ass_new_track.argtypes = [ctypes.c_void_p]

        self.lib.ass_render_frame.restype = ctypes.POINTER(ASS_Image)
        self.lib.ass_render_frame.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int]

        self.lib.ass_set_frame_size.restype = None
        self.lib.ass_set_frame_size.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

        self.lib.ass_process_data.restype = None
        self.lib.ass_process_data.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]

        self.lib.ass_set_fonts.restype = None
        self.lib.ass_set_fonts.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]

        self.lib.ass_free_track.restype = None
        self.lib.ass_free_track.argtypes = [ctypes.c_void_p]

        self.lib.ass_renderer_done.restype = None
        self.lib.ass_renderer_done.argtypes = [ctypes.c_void_p]

        self.lib.ass_library_done.restype = None
        self.lib.ass_library_done.argtypes = [ctypes.c_void_p]

    def __init__(self, width=1920, height=1080, font_family="Helvetica"):
        try:
            self.lib = ctypes.CDLL(LIBASS_PATH)
        except OSError:
            print(f"Could not load libass at {LIBASS_PATH}")
            raise

        self._setup_signatures()

        self.library = self.lib.ass_library_init()
        self.renderer = self.lib.ass_renderer_init(self.library)

        # Initialize fonts (Using CoreText provider on macOS by passing 1)
        self.lib.ass_set_fonts(self.renderer, None, font_family.encode('utf-8'), 1, None, 1)
        self.lib.ass_set_frame_size(self.renderer, width, height)

    def _setup_signatures(self):
        # Define Function Signatures
        self.lib.ass_library_init.restype = ctypes.c_void_p
        self.lib.ass_library_init.argtypes = []

        self.lib.ass_renderer_init.restype = ctypes.c_void_p
        self.lib.ass_renderer_init.argtypes = [ctypes.c_void_p]

        self.lib.ass_new_track.restype = ctypes.c_void_p
        self.lib.ass_new_track.argtypes = [ctypes.c_void_p]

        self.lib.ass_render_frame.restype = ctypes.POINTER(ASS_Image)
        self.lib.ass_render_frame.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int]

        self.lib.ass_set_frame_size.restype = None
        self.lib.ass_set_frame_size.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

        self.lib.ass_process_data.restype = None
        self.lib.ass_process_data.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]

        self.lib.ass_set_fonts.restype = None
        self.lib.ass_set_fonts.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]

        self.lib.ass_free_track.restype = None
        self.lib.ass_free_track.argtypes = [ctypes.c_void_p]

        self.lib.ass_renderer_done.restype = None
        self.lib.ass_renderer_done.argtypes = [ctypes.c_void_p]

        self.lib.ass_library_done.restype = None
        self.lib.ass_library_done.argtypes = [ctypes.c_void_p]

    def close(self):
        self.lib.ass_renderer_done(self.renderer)
        self.lib.ass_library_done(self.library)
    
    def _get_glyph_rects(self, text, style_config, wrap_style=0):
        track = self.lib.ass_new_track(self.library)
        # Construct a minimal ASS script for rendering
        ass_script = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: {wrap_style}

{style_config}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:05.00,Default,,0,0,0,,{text}
"""
        encoded_ass = ass_script.encode('utf-8')
        self.lib.ass_process_data(track, encoded_ass, len(encoded_ass))

        # Render at 1000ms (1s)
        img_list = self.lib.ass_render_frame(self.renderer, track, 1000, 0)
        if not img_list:
            self.lib.ass_free_track(track)
            return []

        rects = []
        current = img_list
        while current:
            img = current.contents
            if img.w > 0 and img.h > 0:
                # Check alpha in color. libass color is ØxRRGGBBAA (big endian) or similar.
                # The last byte is Alpha. 0xFF is transparent.
                alpha = img.color & 0xFF

                if alpha == 0xFF:
                        # Skip fully transparent images
                        current = img.next
                        continue

                # Scan bitmap to find actual content bounds
                bitmap_size = img.h * img.stride
                buf = ctypes.string_at(img.bitmap, bitmap_size)
                
                content_min_x = img.w
                content_max_x = -1
                content_min_y = img.h
                content_max_y = -1
                
                has_content = False
                
                for y in range(img.h):
                    row_start = y * img.stride
                    row = buf[row_start : row_start + img.w]
                    
                    if not any(row):
                        continue
                        
                    if not has_content:
                        content_min_y = y
                    content_max_y = y
                    has_content = True
                    
                    # Find min x in this row
                    for x in range(img.w):
                        if row[x] > 0:
                            if x < content_min_x: content_min_x = x
                            break
                            
                    # Find max x in this row
                    for x in range(img.w - 1, -1, -1):
                        if row[x] > 0:
                            if x > content_max_x: content_max_x = x
                            break
                
                if has_content:
                    global_min_x = img.dst_x + content_min_x
                    global_min_y = img.dst_y + content_min_y
                    global_max_x = img.dst_x + content_max_x + 1
                    global_max_y = img.dst_y + content_max_y + 1
                    rects.append((global_min_x, global_min_y, global_max_x, global_max_y))

            current = img.next

        self.lib.ass_free_track(track)
        return rects

    def measure_text(self, text, style_config, wrap_style=0, line_height=None):
        rects = self._get_glyph_rects(text, style_config, wrap_style)
        if not rects:
            return None

        min_x = min(r[0] for r in rects)
        min_y = min(r[1] for r in rects)
        max_x = max(r[2] for r in rects)
        max_y = max(r[3] for r in rects)
        
        height = int(max_y - min_y)
        if line_height is not None:
            height = line_height

        return {
            "x": int(min_x),
            "y": int(min_y) ,
            "width": int(max_x - min_x) + 1,
            "height": height
        }

    def measure_lines(self, text, style_config, wrap_style=0):
        rects = self._get_glyph_rects(text, style_config, wrap_style)
        if not rects: return []
        
        # Cluster rects into lines
        # Sort by Y (top)
        rects.sort(key=lambda r: r[1])
        
        lines = []
        if not rects: return []
        
        current_line = [rects[0]]
        
        for r in rects[1:]:
            l_min_y = min(x[1] for x in current_line)
            l_max_y = max(x[3] for x in current_line)
            l_height = l_max_y - l_min_y
            
            r_min_y = r[1]
            r_max_y = r[3]
            r_height = r_max_y - r_min_y
            
            overlap_start = max(l_min_y, r_min_y)
            overlap_end = min(l_max_y, r_max_y)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > 0.4 * min(l_height, r_height):
                 current_line.append(r)
            else:
                 lines.append(current_line)
                 current_line = [r]
                 
        lines.append(current_line)
        
        results = []
        for line in lines:
            min_x = min(r[0] for r in line)
            min_y = min(r[1] for r in line)
            max_x = max(r[2] for r in line)
            max_y = max(r[3] for r in line)
            results.append({
                "x": int(min_x),
                "y": int(min_y),
                "width": int(max_x - min_x) + 1,
                "height": int(max_y - min_y)
            })
            
        return results

def strip_tags(text):
    return re.sub(r"\{.*?\}", "", text)

def create_rounded_box_path(w, h, r):
    # Top-Left at 0,0
    # Top-Right: w, 0
    # Bottom-Right: w, h
    # Bottom-Left: 0, h

    k = 0.5522847498

    x_left = 0
    x_right = w
    y_top = 0
    y_bottom = h

    # Start at top-left (after corner)
    path = f"m {int(x_left + r)} {int(y_top)} "

    # Top edge
    path += f"l {int(x_right - r)} {int(y_top)} "

    # Top-Right corner
    # b cp1x cply cp2x cp2y endx endy
    cp1x = x_right - r + (k * r)
    cp1y = y_top
    cp2x = x_right
    cp2y = y_top + r - (k * r)
    endx = x_right
    endy = y_top + r
    path += f"b {int(cp1x)} {int(cp1y)} {int(cp2x)} {int(cp2y)} {int(endx)} {int(endy)}"

    # Right edge
    path += f"l {int(x_right)} {int(y_bottom - r)} "

    # Bottom-Right corner
    cp1x = x_right
    cp1y = y_bottom - r + (k * r)
    cp2x = x_right - r + (k * r)
    cp2y = y_bottom
    endx = x_right - r
    endy = y_bottom
    path += f"b {int(cp1x)} {int(cp1y)} {int(cp2x)} {int(cp2y)} {int(endx)} {int(endy)} "

    # Bottom edge
    path += f"l {int(x_left + r)} {int(y_bottom)} "

    # Bottom-Left corner
    cp1x = x_left + r - (k * r)
    cp1y = y_bottom
    cp2x = x_left
    cp2y = y_bottom - r + (k * r)
    endx = x_left
    endy = y_bottom - r
    path += f"b {int(cp1x)} {int(cp1y)} {int(cp2x)} {int(cp2y)} {int(endx)} {int(endy)} "

    # Left edge
    path += f"l {int(x_left)} {int(y_top + r)}"

    # Top-Left corner
    cp1x = x_left
    cp1y = y_top + r - (k * r)
    cp2x = x_left + r - (k * r)
    cp2y = y_top
    endx = x_left + r
    endy = y_top
    path += f"b {int(cp1x)} {int(cp1y)} {int(cp2x)} {int(cp2y)} {int(endx)} {int(endy)} "

    return path

def create_squared_box_path(w, h):
    """
    Creates a simple squared box path without rounded corners.
    
    Parameters:
    w: width of the box
    h: height of the box
    
    Returns:
    A path string suitable for ASS drawing mode
    """
    # Top-Left at 0,0
    # Top-Right: w, 0
    # Bottom-Right: w, h
    # Bottom-Left: 0, h
    
    x_left = 0
    x_right = w
    y_top = 0
    y_bottom = h
    
    # Start at top-left
    path = f"m {int(x_left)} {int(y_top)} "
    
    # Top edge to top-right
    path += f"l {int(x_right)} {int(y_top)} "
    
    # Right edge to bottom-right
    path += f"l {int(x_right)} {int(y_bottom)} "
    
    # Bottom edge to bottom-left
    path += f"l {int(x_left)} {int(y_bottom)} "
    
    # Left edge back to top-left
    path += f"l {int(x_left)} {int(y_top)}"
    
    return path

def time_to_ms(time_str):
    h, m, s = time_str.split(':')
    s, cs = s.split('.')
    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(cs) * 10

def process_subtitles(input_file, output_file, box_padding_x=50, box_padding_y=50, box_radius=20, box_color="FF5F0F", style="bullet"):
    # Configuration
    PLAY_RES_X = 1920
    PLAY_RES_Y = 1080

    # Box styling
    BOX_PADDING_X = box_padding_x
    BOX_PADDING_Y = box_padding_y
    BOX_RADIUS = box_radius
    BOX_COLOR = f"&H{box_color}&"
    BOX_ALPHA = "&H00&" # Opaque
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Extract Styles Header and find font
    style_lines = []
    in_styles = False
    font_name = "Helvetica" # Default fallback
    outline = 0
    shadow = 0
    style_margin_v = 0
    
    for line in lines:
        if line.strip() == "[V4+ Styles]":
            in_styles = True
            style_lines.append(line)
            continue
        if in_styles:
            if line.startswith("["): #Next section
                in_styles = False
            else:
                style_lines.append(line)
                if line.startswith("Style:"):
                    # Style: Default,Helvetica,130, ...
                    parts = line.split(",")
                    if len(parts) > 1:
                        # Assuming standard format where Name is 0 and Fontname is 1
                        # We strip "Style: " from the first part
                        p0 = parts[0].replace("Style:", "").strip()
                        if p0 == "Default":
                            font_name = parts[1].strip()
                            try:
                                outline = float(parts[16].strip())
                                shadow = float(parts[17].strip())
                                style_margin_v = int(parts[21].strip())
                            except (IndexError, ValueError):
                                pass

    style_config ="".join(style_lines)

    # Initialize Measurer
    try:
      measurer = LibassMeasurer(PLAY_RES_X, PLAY_RES_Y, font_family=font_name)
    except Exception as e:
        print(f"Error initializing libass: {e}")
        return

    new_lines = []
    events_started = False

    # Regex to find the highlighted word block
    # Looking for: prefix {tags\highlight} word {\r} suffix
    # The tag block must contain \highlight
    pattern = re.compile(r"^(.*?)(\{.*?\\highlight.*?\})(.*?)(\{\\r\})(.*)$")

    for line in lines:
        line = line.strip()
        if line.startswith("[Events]"):
            events_started = True
            new_lines.append(line)
            continue

        if not events_started or not line.startswith("Dialogue:"):
            new_lines.append(line)
            continue

        # Parse Dialogue line
        # Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
        parts = line.split(",", 9)
        if len(parts) < 10:
            new_lines.append(line)
            continue

        layer = parts[0].split(":")[1].strip()
        start_time = parts[1]
        end_time = parts[2]
        start_ms = time_to_ms(start_time)
        end_ms = time_to_ms(end_time)
        duration_ms = end_ms - start_ms
        style_name = parts[3]
        name = parts[4]
        margin_l = int(parts[5])
        margin_r = int(parts[6])
        margin_v = int(parts[7])
        effect = parts[8]
        text = parts[9]

        match = pattern.match(text)
        if match:
            prefix_raw = match.group(1)
            tag_block = match.group(2)
            word_raw = match.group(3)
            reset_tag = match.group(4)
            suffix_raw = match.group(5)

            if style in ["line", "roundline"]:
                # Measure lines of the FULL text
                line_bboxes = measurer.measure_lines(text, style_config, 0)
                
                for bbox in line_bboxes:
                    box_w = bbox['width'] + BOX_PADDING_X
                    box_h = bbox['height'] + BOX_PADDING_Y
                    
                    box_center_x = bbox['x'] + (box_w / 2) - (BOX_PADDING_X / 2)
                    box_center_y = bbox['y'] + (box_h / 2) - (BOX_PADDING_Y / 2)
                    
                    if style == "roundline":
                        path = create_rounded_box_path(box_w, box_h, BOX_RADIUS)
                    else:
                        path = create_squared_box_path(box_w, box_h)
                    
                    box_tags = (
                        f"{{\\an5}}" # Align 5 (Center)
                        f"{{\\p1}}" # Drawing mode 1
                        f"{{\\1c{BOX_COLOR}}}" # Primary color
                        f"{{\\1a{BOX_ALPHA}}}" # Alpha
                        f"{{\\bord0\\shad0}}" # No border/shadow
                        f"{{\\pos({int(box_center_x)},{int(box_center_y)})}}" # Position at center
                    )
                    
                    box_event = f"Dialogue: 0,{start_time},{end_time},{style_name},{name},{margin_l},{margin_r},{margin_v},{effect},{box_tags}{path}{{\\p0}}"
                    new_lines.append(box_event)
            else:
                # Construct Masked Text for Measurement
                # We want to measure the position of 'word_raw' within the full text.
                # We make everything else transparent.
                # Note: We need to handle existing tags in prefix/suffix if they affect layout?
                # Ideally, we keep the structure but inject alpha tags.
                # However, injecting alpha tags into existing tags is tricky.
                # Simplification: Assume prefix/suffix don't have complex alpha animations.
                # We wrap prefix and suffix in \alpha&HFF&.
                # But we must ensure we don't override existing style overrides that might be important for layout (like font size).
                # If we just append {\alpha&HFF&), it overrides previous alpha.

                # Strategy:
                # 1. Prefix: {\alpha&HFF&} + prefix_raw
                # 2. Words: {\alpha&H00&) + word_raw (ensure it's visible)
                # 3. Suffix: {\alpha&HFF&} + suffix_raw

                # Wait, if prefix_raw contains {\r}, it resets styles.
                # If prefix_raw contains overrides, we need to make sure our alpha applies.
                # Appending {\alpha&HFF&) at the start of prefix might be overridden by later tags in prefix.
                # But usually tags are at the start.
                # A safer way is to replace all text content in prefix with transparent text? No, that changes layout.

                # Let's try simply wrapping.
                # We need to ensure the word is opaque.

                # Also, we need to strip the \highlight tag from the tag_ block because it's not a standard ASS tag
                # and might confuse libass or just be ignored. It's likely a custom tag for this script.
                # The user's regex implies it's there.

                # Clean tag block for rendering (remove \highlight if it's not valid ASS, but libass ignores unknown tags usually)
                # We'll just use the text as is but inject alphas.

                # 1) Measure whole line once, to get a consistent height for this dialogue
                line_bbox = measurer.measure_text(text, style_config, 2) #, None, yadjust, style_margin_v)
                line_bbox_wrapped = measurer.measure_text(text, style_config, 0)

                masked_text = (
                    r"{\alpha&HFF&}" + prefix_raw +
                    r"{\alpha&H00&}" + word_raw +
                    r"{\alpha&HFF&}" + suffix_raw
                )

                # Measure
                bbox = measurer.measure_text(masked_text, style_config)
                
                if bbox: # and word_bbox:
                    ref_bbox = line_bbox or bbox

                    # Box dimensions
                    box_w = bbox['width'] + BOX_PADDING_X
                    box_h = ref_bbox['height'] + BOX_PADDING_Y
                                
                    # Box Center
                    box_center_x = bbox['x'] + (box_w / 2) - (BOX_PADDING_X / 2)
                    box_center_y = ref_bbox["y"] + (box_h / 2) - (BOX_PADDING_Y / 2)

                    if (line_bbox_wrapped['height'] > line_bbox['height']):
                        if bbox['y'] + 1 < line_bbox['y']:
                            box_center_y =  line_bbox_wrapped['y'] + (box_h/ 2) - (BOX_PADDING_Y / 2)

                    # Debug info
                    print(f"Word: '{word_raw}' | BBox: {bbox} | Center: ({int(box_center_x)}, {int(box_center_y)})")

                    if style == "bullet":
                        path = create_rounded_box_path(box_w, box_h, BOX_RADIUS)
                    else: # box
                        path = create_squared_box_path(box_w, box_h)

                    t_pop_end = 150
                    t_depop_start = max(t_pop_end, duration_ms - 150)

                    box_tags = (
                        f"{{\\an5}}" # Align 5 (Center)
                        f"{{\\p1}}" # Drawing mode 1
                        f"{{\\1c{BOX_COLOR}}}" # Primary color
                        f"{{\\1a{BOX_ALPHA}}}" # Alpha
                        f"{{\\bord0\\shad0}}" # No border/shadow
                        f"{{\\pos({int(box_center_x)},{int(box_center_y)})}}" # Position at center
                        f"{{\\fscx100\\fscy100}}"
                        f"{{\\t(0,{t_pop_end},\\fscx120\\fscy120)}}"
                        f"{{\\t({t_depop_start},{duration_ms},\\fscx100\\fscy100)}}"
                    )

                    box_event = f"Dialogue: 0,{start_time},{end_time},{style_name},{name},{margin_l},{margin_r},{margin_v},{effect},{box_tags}{path}{{\\p0}}"
                    new_lines.append(box_event)
                else:
                    print(f"Could not measure word: {word_raw}")

            # Update original Line to Layer 1
            # We should remove the \highlight tag from the output text if it's not needed anymore,
            # or keep it if the user wants it. The regex captured it in tag_block.
            # Let's keep the original text line exactly as is, just layer 1.

            original_line_layer_1= f"Dialogue: 1,{start_time},{end_time},{style_name},{name},{margin_l},{margin_r},{margin_v},{effect},{text}"
            new_lines.append(original_line_layer_1)

        else:
            parts[0] = "Dialogue: 1"
            new_lines.append(",".join(parts))

    measurer.close()

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(new_lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate background boxes for subtitles.")
    parser.add_argument("input_file", help="Input ASS file path")
    parser.add_argument("output_file", help="Output ASS file path")
    parser.add_argument("--padding_x", type=int, default=40, help="Horizontal padding for the box")
    parser.add_argument("--padding_y", type=int, default=40, help="Vertical padding for the box")
    parser.add_argument("--radius", type=int, default=20, help="Corner radius for the box")
    parser.add_argument("--color", type=str, default="FF5F0F", help="Color of the box in BBGGRR hex format")
    parser.add_argument("--style", type=str, default="bullet", choices=["box", "bullet", "line", "roundline"], help="Style of the background box")

    args = parser.parse_args()
    process_subtitles(args.input_file, args.output_file, args.padding_x, args.padding_y, args.radius, args.color, args.style)