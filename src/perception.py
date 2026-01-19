#Perception module for ArUco/QR detection
#input: a signle frame (e.g. a numpy image from OpenCV)
#Processing: run a detector 
#Output: 
    # detected: whether the target was found
    # id / ids: which marker(s) were found
    # corners / pose info (optional): where it is in the image (for tracking / aiming)
    # confidence (optional, proxy): how reliable the detection seems

#perception.py answers “did we detect the target, and where is it?”, and everything else (uncertainty + action) builds on that.

