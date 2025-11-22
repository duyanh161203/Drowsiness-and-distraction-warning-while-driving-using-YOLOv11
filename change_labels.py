import os

# Mapping cũ -> mới
# 0,1,3 => 0 (Distracted)
# 2 => 1 (Distracted Phone)
# 4 => 2 (drinking)
# 5 => 3 (drowsy)
# 6 => 4 (seatbelt)
# 7 => 5 (smoking)
def convert_label(old_id):
    if old_id in [0, 1, 3]:
        return 0
    elif old_id == 2:
        return 1
    elif old_id == 4:
        return 2
    elif old_id == 5:
        return 3
    elif old_id == 6:
        return 4
    elif old_id == 7:
        return 5
    else:
        return None

def update_labels(label_dir):
    for fname in os.listdir(label_dir):
        if not fname.endswith('.txt'):
            continue
        fpath = os.path.join(label_dir, fname)
        new_lines = []
        with open(fpath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                old_id = int(parts[0])
                new_id = convert_label(old_id)
                if new_id is not None:
                    parts[0] = str(new_id)
                    new_lines.append(' '.join(parts))
        with open(fpath, 'w') as f:
            for l in new_lines:
                f.write(l + '\n')

for split in ['train', 'valid', 'test']:
    label_dir = f'd:/DATN/Driver Mentoring.v8i.yolov11/{split}/labels'
    update_labels(label_dir)