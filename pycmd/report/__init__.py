def printEntry(w, qid, ids, dists):
    w.write(f"{qid}")
    for i in range(len(ids)):
        w.write("\t{}:{:f}".format(ids[i], dists[i]))
    w.write("\n")

def printAllNN(w, ids, dists):
    for i in range(len(ids)):
        w.write("{}:{:f}\t".format(ids[i], dists[i]))
    w.write("\n")
