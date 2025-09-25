# perceptron-c
Building a classic 0-9 digit perceptron in C for educational purposes.

## testing 
```bash
# compile from project root
clang -std=c11 -O2 -Wall -Wextra -pedantic src/main.c -o perceptron

# try the CLI
./perceptron --help
./perceptron train
./perceptron predict
./perceptron banana   # should error
```