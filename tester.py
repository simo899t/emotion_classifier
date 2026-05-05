d = {'1': 3, '2': 4}
rev_dict = {}
for key, idx in d.items():
    rev_dict[idx] = key


print(d)

a = d.get('3',0)

print(rev_dict)

print(a)