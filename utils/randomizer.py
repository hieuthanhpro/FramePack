from random import randint

def _get_random_seeds(num=0, _range=(1, 100000), existed=set()):
  ret = []
  while len(ret) < num:
    rand = randint(_range[0], _range[1])
    if rand not in existed:
      ret.append(rand)
  return ret