Face enrollment layout (one folder per person; folder name = on-screen label):

  face_registry/
    爸爸/     <- put several .jpg/.png frontal photos here
    妈妈/
    儿子/

Capture from webcam:
  python scripts/capture_face_samples.py --registry face_registry --name 爸爸 --cam 1
  python scripts/capture_face_samples.py --registry face_registry --name Aaron Xie --cam 1

PowerShell: names with spaces — use either form (no angle brackets):
  --name "Aaron Xie"
  --name Aaron Xie

Which camera index is USB? (Windows)
  python scripts/capture_face_samples.py --list-cams
  Then use e.g. --cam 2 if your external device is index 2.
  Enrollment uses the same backends as the main app (DirectShow / MSMF).

Run the demo:
  python demo.py --perception face --face-registry face_registry --cam 1

Tune --face-threshold if you see "?" too often (try 95) or wrong IDs (try 70).
