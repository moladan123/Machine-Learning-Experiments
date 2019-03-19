from numpy import random
import sys
file = open("Words")
words = file.read().split()
file.close()
# makes a markov chain based on last 2 letters
l = "`abcdefghijklmnopqrstuvwxyz"

# initialize markov chain
letters = {}  # 26 letters as well as space included
for i in range(27):
    letters[chr(ord("`") + i)] = [0.0 for i in range(27)]

# train with the data
for word in words:
    a = "`"
    for i in word + "`": # first letter -> next letter
        try:
            letters[a][ord(i) - ord("`")] += 1.0
        except IndexError:
            pass
        except KeyError:
            pass
        a = i

# square regularize
for i in l:
    letters[i] = [letters[i][j] ** 3 for j in range(27)]

    s = sum(letters[i])
    if s < 0.00000001:
        print(letters[i])
        continue
    letters[i] = [letters[i][j] / s for j in range(27)]
    print(letters[i])

# generates text from previous letter
current_letter = "`"
for i in range(1000):
    new_letter = "".join(random.choice(list(l), 1, p=letters[current_letter]))

    if new_letter == '`':
        print("")
    else:
        sys.stdout.write(new_letter)

    current_letter = new_letter
