# lab-1

Каждая строка входного файла содержиткоординаты вершин разделенные запятой

Пример:

> input.txt
```
0, -2
2, -1
1, 1
-2, 1 
-3, 0s
```

Выходной файл содержит индексы вершин i1 i2 i3 i4, такие что i1 и i1 - 1 лежат на одном ребре obb

Пример:

> output.txt
```
0 1 2 4
```

Команда для запуска:

```console
$ cd lab-1
$ python3 -m lab-1 input.txt -o ouput.txt  -c square
```