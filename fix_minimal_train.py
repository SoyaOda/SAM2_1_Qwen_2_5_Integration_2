#!/usr/bin/env python3
"""Fix the f-string issue in minimal_train.py"""

import re

# Read the file
with open('minimal_train.py', 'r') as f:
    content = f.read()

# Replace the problematic section
old_pattern = r'''            info_text = f"Step: \{step\}
"
            info_text \+= f"Dice Score: \{dice:\.4f\}
"
            info_text \+= f"IoU: \{iou:\.4f\}
"
            info_text \+= f"LM Loss: \{lm_loss:\.4f\}
"
            info_text \+= f"Mask Size: \{gt_mask_np\.shape\}
"
            info_text \+= f"Image Size: \{image_np\.shape\[:2\]\}
"
            info_text \+= f"Label Masking: \{masked_count\} masked, \{unmasked_count\} unmasked

"
            if len\(user_text\) > 200:
                info_text \+= f"User: \{user_text\[:200\]\}\.\.\.

"
            else:
                info_text \+= f"User: \{user_text\}

"
            if len\(assistant_text\) > 100:
                info_text \+= f"Assistant: \{assistant_text\[:100\]\}\.\.\."
            else:
                info_text \+= f"Assistant: \{assistant_text\}"'''

new_pattern = '''            info_text = f"Step: {step}\\n"
            info_text += f"Dice Score: {dice:.4f}\\n"
            info_text += f"IoU: {iou:.4f}\\n"
            info_text += f"LM Loss: {lm_loss:.4f}\\n"
            info_text += f"Mask Size: {gt_mask_np.shape}\\n"
            info_text += f"Image Size: {image_np.shape[:2]}\\n"
            info_text += f"Label Masking: {masked_count} masked, {unmasked_count} unmasked\\n\\n"
            if len(user_text) > 200:
                info_text += f"User: {user_text[:200]}...\\n\\n"
            else:
                info_text += f"User: {user_text}\\n\\n"
            if len(assistant_text) > 100:
                info_text += f"Assistant: {assistant_text[:100]}..."
            else:
                info_text += f"Assistant: {assistant_text}"'''

# Use direct string replacement for the problematic section
# Look for the specific line pattern
lines = content.split('\n')
new_lines = []
i = 0
while i < len(lines):
    if 'info_text = f"Step: {step}' in lines[i]:
        # Skip the problematic lines and insert the fixed version
        new_lines.append('            info_text = f"Step: {step}\\n"')
        new_lines.append('            info_text += f"Dice Score: {dice:.4f}\\n"')
        new_lines.append('            info_text += f"IoU: {iou:.4f}\\n"')
        new_lines.append('            info_text += f"LM Loss: {lm_loss:.4f}\\n"')
        new_lines.append('            info_text += f"Mask Size: {gt_mask_np.shape}\\n"')
        new_lines.append('            info_text += f"Image Size: {image_np.shape[:2]}\\n"')
        new_lines.append('            info_text += f"Label Masking: {masked_count} masked, {unmasked_count} unmasked\\n\\n"')
        
        # Skip ahead to find the if/else block
        while i < len(lines) and 'if len(user_text) > 200:' not in lines[i]:
            i += 1
        
        if i < len(lines):
            new_lines.append('            if len(user_text) > 200:')
            new_lines.append('                info_text += f"User: {user_text[:200]}...\\n\\n"')
            new_lines.append('            else:')
            new_lines.append('                info_text += f"User: {user_text}\\n\\n"')
            
            # Skip to assistant text
            while i < len(lines) and 'if len(assistant_text) > 100:' not in lines[i]:
                i += 1
            
            if i < len(lines):
                new_lines.append('            if len(assistant_text) > 100:')
                new_lines.append('                info_text += f"Assistant: {assistant_text[:100]}..."')
                new_lines.append('            else:')
                new_lines.append('                info_text += f"Assistant: {assistant_text}"')
                
                # Skip to the next non-related line
                while i < len(lines) and 'ax6.text(' not in lines[i]:
                    i += 1
                i -= 1  # Back up one to not skip the ax6.text line
    else:
        new_lines.append(lines[i])
    i += 1

# Write back
with open('minimal_train.py', 'w') as f:
    f.write('\n'.join(new_lines))

print("Fixed minimal_train.py")