import os
import re
import logging
import csv
from tqdm import tqdm

function_list_file = "./data/functions.csv"
src_dir = "/home/wuyuhao/dataset/bigclonebench/files/dataset/"
# ccfx_ext = ".java.2_0_0_0.default.ccfxprep"
cmp_dir = "/home/wuyuhao/dataset/bigclonebench/files/dataset/functions/"
output_dir = "/home/wuyuhao/dataset/bigclonebench/files/dataset/functions2/"
output_ext = ".java"

log_file = os.path.join(src_dir, "failed_functions.txt")
logging.basicConfig(filename=log_file)
logger = logging.getLogger(__name__)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

skip_to = 0


def main():
    """Extract function snippets from the source files based on the line number
    provided by bigclonebench."""

    with open(function_list_file) as f:
        for i, l in enumerate(f):
            pass
        total = i

    with open(function_list_file) as f:
        reader = csv.reader(f)

        # Loop through 1function list
        for i, row in tqdm(enumerate(reader), total=total, desc="Processing function"):

            if i+1 < skip_to:
                continue

            # Get function info
            func_id, name, path, start, end = row[:]
            start = int(start)
            end = int(end)
            # print row

            cmp_path = os.path.join(cmp_dir, func_id + output_ext)

            if os.path.exists(cmp_path):
                continue

            # logger.info("Processing [{0}/{1}]".format(i, total))
            print ("Processing [{0}/{1}]".format(i, total))

            # Locate the token file path
            process_row(func_id, name, path, start, end)

            # break


def process_row(func_id, name, path, start, end):

    src_path = os.path.join(src_dir, path, name)

    if not os.path.exists(src_path):
        logger.warning("func_id:{1}. File not found: {0}".format(src_path, func_id))
        return -1

    # Extract corresponding function lines
    func_snippet = get_lines_from(src_path, start, end)
    # print func_snippet

    if func_snippet is None:
        logger.warning("Snippet in range [{1},{2}] not found in file: {0}".format(src_path, start, end))
        return -2

    # Save the snippet to file
    save_path = os.path.join(output_dir, func_id + output_ext)

    try:
        with open(save_path, 'w') as fo:
            fo.write(func_snippet)
    except IOError as e:
        logger.error("func_id:{0}, file:{1}, msg:{2}".format(func_id, path, e))
        # print("func_id: [{0}], file: [{1}], msg: {2}".format(func_id, name, e))
        return -3

    return 0


def get_lines_from(src_path, start, end):
    # print(start, end, src_path)
    cont = []
    with open(src_path) as f:
        for i, line in enumerate(f):
            if start <= i + 1 <= end:
                # print line
                cont.append(line)
    return "".join(cont)


def test():
    start = 60
    end = 72
    path = '/home/wuyuhao/dataset/bigclonebench/files/dataset/selected/1468334.java'
    func_snippet = get_lines_from(path, start, end)
    print func_snippet


def test2():
    process_row("12810747", "1543652.java", "selected", 231, 239)


if __name__ == '__main__':
    main()
    # test()
    # test2()