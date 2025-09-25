# perceptron-c
Building a classic 0-9 digit perceptron in C for educational purposes.

## testing 
```bash
# compile from project root
clang -std=c11 -O2 -Wall -Wextra -pedantic src/main.c src/data.c -o perceptron

# test datasets
./perceptron train xor
./perceptron train and
./perceptron train or
./perceptron train banana   # should error
```