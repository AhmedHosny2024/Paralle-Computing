// Ahmed_Hosny_Abdelrazik_1#_2
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int columnSum(int **arr, int nrows, int ncols) {
    int sum = 0;
    for (int j = 0; j < ncols; j++) {
        char columnConcat[100] = "";
        for (int i = 0; i < nrows; i++) {
            char num[10];
            sprintf(num, "%d", arr[i][j]);
            strcat(columnConcat, num);
        }
        sum += atoi(columnConcat);
    }
    return sum;
}

int main(int argc, char *argv[]) {

    int nrows = atoi(argv[1]);
    int ncols = atoi(argv[2]);
    int **arr = (int **)malloc(nrows * sizeof(int *));
    if (arr == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }
    int count = 3;
    for (int i = 0; i < nrows; i++) {
        arr[i] = (int *)malloc(ncols * sizeof(int));
        if (arr[i] == NULL) {
            printf("Memory allocation failed\n");
            return 1;
        }
        for (int j = 0; j < ncols; j++) {
            arr[i][j] = atoi(argv[count]);
            count++;
        }
    }
    int sum = columnSum(arr, nrows, ncols);

    printf("Sum : %d\n", sum);

    for (int i = 0; i < nrows; i++) {
        free(arr[i]);
    }
    free(arr);

    return 0;
}
