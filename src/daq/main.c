#include "daq.h"

#include <errno.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 7 || strcmp(argv[1], "--input") != 0 ||
        strcmp(argv[3], "--output") != 0 || strcmp(argv[5], "--stats") != 0) {
        fprintf(stderr, "usage: daq --input FILE|- --output FILE|- --stats FILE\n");
        return 2;
    }
    FILE *input = strcmp(argv[2], "-") == 0 ? stdin : fopen(argv[2], "rb");
    if (input == NULL) {
        fprintf(stderr, "input: %s\n", strerror(errno));
        return 2;
    }
    FILE *output = strcmp(argv[4], "-") == 0 ? stdout : fopen(argv[4], "w");
    FILE *metrics = fopen(argv[6], "w");
    if (output == NULL || metrics == NULL) {
        fprintf(stderr, "output or stats: %s\n", strerror(errno));
        if (input != stdin) fclose(input);
        if (output != NULL && output != stdout) fclose(output);
        if (metrics != NULL) fclose(metrics);
        return 2;
    }
    daq_stats stats;
    int result = daq_process(input, output, &stats);
    if (daq_write_stats(metrics, &stats) != 0) result = -1;
    if (input != stdin && fclose(input) != 0) result = -1;
    if (output != stdout && fclose(output) != 0) result = -1;
    if (fclose(metrics) != 0) result = -1;
    if (result != 0) fprintf(stderr, "acquisition I/O error\n");
    return result == 0 ? 0 : 1;
}
