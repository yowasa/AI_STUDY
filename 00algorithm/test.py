from scipy.special import perm, comb


def cal(cardNum, effectNum, dropCard):
    return 1 - comb(effectNum, 0) * comb(cardNum - effectNum, dropCard) / comb(cardNum, dropCard)


print(cal(40, 14, 5))
