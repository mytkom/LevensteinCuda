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
Alphabet ACTG:
1. Easy example - easy to follow <10 characters each string 
2. 1k example - 1k characters each string 
3. 10k example - 10k characters each string 
4. 20k example - 20k characters each string 
5. 30k example - 30k characters each string 

There is also an ascii alphabet defined in assets
Alphabet ascii:
1. 20k example - 20k characters each string 

## Usage
```sh
levcuda [path to target string] [path to source string] (-v|-a arg)
```
If -v (verbose) specified as third argument program prints distance matrices and input strings.
If -a (alphabet) is specified and followed by filepath in next arg, it sets custom alphabet for execution
