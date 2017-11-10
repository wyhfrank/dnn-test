import os
import re
import logging
import csv
from tqdm import tqdm


function_list_file = "./data/functions.csv"
src_dir = "/home/wuyuhao/dataset/bigclonebench/files/dataset/"
ccfx_ext = ".java.2_0_0_0.default.ccfxprep"
output_dir = "/home/wuyuhao/dataset/bigclonebench/files/dataset/functions/"
output_ext = ".ccfxprep"

log_file = os.path.join(src_dir, "failed_functions.txt")
logging.basicConfig(filename=log_file)
logger = logging.getLogger(__name__)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def main():
    """Extract tokens of the snippet from the ccfxprep file based on the line number
    provided by bigclonebench. This approach is not accurate since bigclonebench
    treat LF as newline, while CCFinder treats CR+LF as newline."""

    with open(function_list_file) as f:
        reader = csv.reader(f)

        # Loop through 1function list
        for i, row in tqdm(enumerate(reader), desc="Processing function"):

            # Get function info
            func_id, name, path, start, end = row[:]
            # print row

            # Locate the token file path
            token_path = os.path.join(src_dir, path, name + ccfx_ext)

            if not os.path.exists(token_path):
                logger.warning("func_id:{1}. File not found: {0}".format(token_path, func_id))
                continue

            # Extract corresponding token lines
            func_tokens = get_lines_from(token_path, start, end)

            if func_tokens is None:
                logger.warning("Token range [{1},{2}] not found in file: {0}".format(token_path, start, end))
                continue

            # Save the snippet to file
            save_path = os.path.join(output_dir, func_id + output_ext)

            with open(save_path, 'w') as fo:
                fo.write(func_tokens)
            # break


def get_lines_from(token_path, start, end):
    s = decimal2hex(start)
    e = decimal2hex(end)
    # print(s, e, token_path)
    with open(token_path) as f:
        content = f.read()

        m = re.search(r"^{0}(.+\n|\r\n?)*^{1}.+".format(s, e), content, re.MULTILINE)
        # m = re.search(r"(^{0}.+(?:\n|\r\n?)((?:(?:\n|\r\n?).+)+)^{1}.+)".format(s, e), content, re.MULTILINE)
        if m:
            return m.group(0)
        else:
            return None


def decimal2hex(d):
    res = hex(int(d))
    if res.startswith("0x"):
        res = res[2:]
    return res


def test():
    start = 60
    end = 72
    path = '/home/wuyuhao/dataset/bigclonebench/files/dataset/selected/1468334.java.java.2_0_0_0.default.ccfxprep'
    get_lines_from(path, start, end)


if __name__ == '__main__':
    main()
    # test()
