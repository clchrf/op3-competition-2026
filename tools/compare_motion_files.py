#!/usr/bin/env python
# Compare OP3 action files (motion_4095.bin) page by page.
# One file = 256 pages x 512 bytes. Page = 64 byte header + 7 steps x 64 bytes.
# Header: name = bytes 0..13, step count = byte 20 (see op3_action_module action_file_define.h).
# A page is "SAME" when all bytes after the name are equal in every file.
#
# Use:
#   python compare_motion_files.py yuAn=a.bin tfira=b.bin robot=c.bin            (all pages that differ)
#   python compare_motion_files.py --pages 1,50,202 yuAn=a.bin tfira=b.bin       (only these pages)
#   python compare_motion_files.py --all ...                                      (also pages that are the same)
# Works with python 2 and 3.
import sys
import hashlib

PAGE_SIZE = 512
NUM_PAGES = 256


def read_page(data, num):
    p = bytearray(data[num * PAGE_SIZE:(num + 1) * PAGE_SIZE])
    name = bytes(p[0:14]).split(b'\0')[0].decode('latin1').strip()
    steps = p[20]
    body_hash = hashlib.md5(bytes(p[14:])).hexdigest()[:6]
    return name, steps, body_hash


def main(argv):
    pages = None
    show_all = False
    files = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--pages':
            pages = [int(x) for x in argv[i + 1].split(',')]
            i += 2
            continue
        if a == '--all':
            show_all = True
        elif '=' in a:
            label, path = a.split('=', 1)
            files.append((label, path))
        i += 1
    if len(files) < 2:
        print(__doc__ if __doc__ else 'need at least 2 files: label=path')
        return 1

    data = {}
    for label, path in files:
        data[label] = open(path, 'rb').read()
        if len(data[label]) != PAGE_SIZE * NUM_PAGES:
            print('WARNING: %s has %d bytes (expected %d)' % (path, len(data[label]), PAGE_SIZE * NUM_PAGES))

    labels = [f[0] for f in files]
    print('page  same? | ' + ' | '.join(labels))
    print('            (cell = name / steps / hash of the page data; "-" = no name)')
    for num in (pages if pages is not None else range(NUM_PAGES)):
        row = [read_page(data[l], num) for l in labels]
        same = len(set(r[2] for r in row)) == 1
        if same and not show_all and pages is None:
            continue
        cells = ['%s / %d / %s' % (r[0] or '-', r[1], r[2]) for r in row]
        print('%4d  %s | %s' % (num, 'SAME' if same else 'DIFF', ' | '.join(cells)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
