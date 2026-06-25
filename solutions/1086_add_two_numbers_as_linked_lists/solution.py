def add_two_numbers(l1, l2):
    if l1 == [0]:
        return l2
    elif l2 == [0]:
        return l1
    l1_str, l2_str = list(map(str, l1)), list(map(str, l2))
    l1_str, l2_str = "".join(l1_str), "".join(l2_str)
    addition_str = str(int(l1_str) + int(l2_str))
    reverse = addition_str[::-1]
    return list(map(int, list(reverse)))