import csv
import os

# SQL statements for creating the input lists:
# select function_id_one as f1, function_id_two as f2, similarity_line, similarity_token from false_positives;
# select function_id_one as f1, function_id_two as f2, syntactic_type, similarity_line, similarity_token, min_tokens, max_tokens from clones;

clone_list_file = '/home/wuyuhao/dataset/bigclonebench/db/output/clones.csv'
# clone_list_file = '/home/wuyuhao/dataset/bigclonebench/db/output/clones-small.csv'
fp_list_file = '/home/wuyuhao/dataset/bigclonebench/db/output/false_positives.csv'
# fp_list_file = '/home/wuyuhao/dataset/bigclonebench/db/output/false_positives-small.csv'
output_dir = 'output/dataset/'
filename_template = 'dataset.sim-{0}.diff-{1}.csv'

NEED_HEADER = False


# min_token_similarity = 0.3  # 0.3, 0.5, 0.7
# threshold_max_difference = 0.8  # 0.8, 0.9


def make_checker(min_token_similarity, max_token_count_difference):
    def threshold_checker(row):
        sim_token, min_token, max_token = map(float, row[4:7])
        similarity_ok = sim_token >= min_token_similarity
        difference_ok = ((max_token - min_token) / max_token) <= max_token_count_difference
        # print(similarity_ok and difference_ok
        return similarity_ok and difference_ok

    return threshold_checker


def generate_dataset(min_token_similarity, max_token_count_difference):
    output_filename = filename_template.format(min_token_similarity, max_token_count_difference)
    output_file = os.path.join(output_dir, output_filename)
    print("Generating {}".format(output_filename))

    with open(output_file, 'wb') as fo:
        writer = csv.writer(fo)

        # Clones
        with open(clone_list_file, 'rb') as fi:
            reader = csv.reader(fi)
            header = next(reader)
            selected_header = header[0:2] + ['is_clone']
            if NEED_HEADER:
                # print(selected_header)
                writer.writerow(selected_header)

            threshold_checker = make_checker(min_token_similarity, max_token_count_difference)
            rows = filter(threshold_checker, reader)
            for row in rows:
                selected_cols = row[0:2] + [1]
                # print(selected_cols)
                writer.writerow(selected_cols)

        # False positives
        with open(fp_list_file, 'rb') as fi:
            reader = csv.reader(fi)
            header = next(reader)
            for row in reader:
                selected_cols = row[0:2] + [0]
                # print(selected_cols)
                writer.writerow(selected_cols)


def test(p1, p2):
    print(p1, p2)


def main():
    thresholds = [
        (0.1, 0.9),
        (0.5, 0.8),
        (0.7, 0.7),
    ]
    for t in thresholds:
        # test(*t)
        generate_dataset(*t)


if __name__ == '__main__':
    main()
