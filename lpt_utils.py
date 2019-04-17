# Standard parameters of license plate in pixel
lpt_std_size = (69,48)
char_size = (9,21)
Cx = [10,2,10+1,5]
Cy = [1,4]
Dx = [3,3,9]

# The upper-left corner of characters
def L1L2():
    L1 = [(Cx[0],Cy[0])]
    L1.append((L1[0][0] + char_size[0] + Cx[1], Cy[0]))
    L1.append((L1[1][0] + char_size[0] + Cx[2], Cy[0]))
    L1.append((L1[2][0] + char_size[0] + Cx[1], Cy[0]))
    L1.append((Cx[0], Cy[0] + char_size[1] + Cy[1]))
    L1.append((L1[4][0] + char_size[0] + Cx[3], L1[4][1]))
    L1.append((L1[5][0] + char_size[0] + Cx[3], L1[4][1]))
    L1.append((L1[6][0] + char_size[0] + Cx[3], L1[4][1]))
    L2 = [(Dx[0], Cy[0] + char_size[1] + Cy[1])]
    L2.append((L2[0][0] + char_size[0] + Dx[1], L2[0][1]))
    L2.append((L2[1][0] + char_size[0] + Dx[1], L2[0][1]))
    L2.append((L2[2][0] + char_size[0] + Dx[2], L2[0][1]))
    L2.append((L2[3][0] + char_size[0] + Dx[1], L2[0][1]))
    return L1, L2
    
def lptnum_of(filename, has_prefix=True, sep="-"):
    idx = [1,2] if has_prefix else [0,1]
    lptnum = ('.').join(filename.split(".")[:-1])
    if sep == "-":
        lptnum = lptnum.split("-")[0]
    lptnum = lptnum.split("_")
    return lptnum[idx[0]] + "_" + lptnum[idx[1]]