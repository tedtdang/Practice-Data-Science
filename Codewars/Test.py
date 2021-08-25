def to_shift(cur_row, cur_col, moves):
    moves += 1 + 5 - cur_row
    moves += cur_col + 1
    cur_row = 5
    cur_col = 0
    print(f'In to_shift: moves = {moves}')



def to_space(cur_row, cur_col, moves):
    moves += 1 + 5 - cur_row
    moves += 1 + abs(1 - cur_col)
    cur_row = 5
    cur_col = 1
    print(f'In to_space: moves = {moves}')
    return moves, cur_row, cur_col


def to_char(cur_row, cur_col, moves, char, table, upper):
    print(f'Char: {char}')
    if not upper:
        if char.isupper():
            moves, cur_row, cur_col = to_shift(cur_row, cur_col, moves)
            print(f'In not upper and char is upper: {moves}, {cur_row}, {cur_col}')
            upper = True
            char = char.lower()
        for i, v in enumerate(table):
            if char in v:
                moves += 1 + abs(i - cur_row)
                moves += abs(v.index(char) - cur_col)
                print(f'Moves in to_char {moves}')
                cur_row = i
                cur_col = v.index(char)
                break
    else:
        if char.islower():
            moves, cur_row, cur_col = to_shift(cur_row, cur_col, moves)
            upper = False
        for i, v in enumerate(table):
            if char in v:
                moves += 1 + abs(i - cur_row)
                moves += abs(v.index(char) - cur_col)
                print(f'Moves in to_char {moves}')
                cur_row = i
                cur_col = v.index(char)
                break
    return moves, cur_row, cur_col

def tv_remote(word):
    cur_row = cur_col = moves = 0
    table = ["abcde123", "fghij456", "klmno789", "pqrst.@0", "uvwxyz_/"]
    up = False
    for char in word:
        if char.isspace():
            moves, cur_row, cur_col = to_space(cur_row, cur_col, moves)
        else:
            moves, cur_row, cur_col = to_char(cur_row, cur_col, moves, char, table, up)
    return moves


print(tv_remote('CodeWars'))
