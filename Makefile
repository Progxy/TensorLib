FLAGS = -std=c11 -Wall -Wextra -pedantic

build: main.c
	gcc $(FLAGS) main.c -o demo
	./out/ml

debug: main.c
	gcc $(FLAGS) -ggdb main.c -o demo
	gdb ./out/ml