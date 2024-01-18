# LevensteinCuda

## Build
To build binary use makefile:
```sh
make
```
To clean binary and transformation files:
```sh
make clean
```

## Assets
There are two pairs of assets in /assets:
1. Long example - 10000 characters each string
1. Short example - <10 characters each string

## Usage
```sh
levcuda [path to target string] [path to source string] (-v)
```
If -v (verbose) specified as third argument program prints distance matrices and input strings
