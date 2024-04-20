FLAGS = -std=c11 -Wall -Wextra -pedantic

build: main.c
	gcc $(FLAGS) main.c -o out/demo
	./out/demo

debug: main.c
	gcc $(FLAGS) -ggdb main.c -o out/demo
	gdb ./out/demo