# HW1: Search

## Implement:
Write your search algorithms in *search.py* and do not edit any other files, except for testing.

## Requirements:
```
python3
pygame
```
## Running:
The main file to run this homework is hw1.py:

```
usage: hw1.py [-h] [--method {bfs,astar,astar_corner,astar_multi,fast}] [--scale SCALE]
              [--fps FPS] [--human] [--save SAVE]
              filename
```

Examples of how to run HW1:
```
python hw1.py bigMaze.txt --method bfs
```
```
python hw1.py tinySearch.txt --scale 30 --fps 10 --human
```

For help run:
```
python hw1.py -h
```
Help Output:
```
HW1 Search

positional arguments:
  filename              path to maze file [REQUIRED]

optional arguments:
  -h, --help            show this help message and exit
  --method {bfs,astar,astar_corner,astar_multi,fast}
                        search method - default bfs
  --scale SCALE         scale - default: 20
  --fps FPS             fps for the display - default 30
  --human               flag for human playable - default False
  --save SAVE           save output to image file - default not saved
```
