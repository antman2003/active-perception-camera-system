/*
 * Minimal pan/tilt Arduino firmware for Session 16.
 *
 * Features:
 * - On boot: attach servos, center, run a small self-test motion.
 * - Serial protocol: "P90 T90" (newline terminated).
 * - Partial updates are allowed: "P100" or "T80".
 * - Replies with READY / OK / ERR for easier PC-side debugging.
 */

#include <Servo.h>

namespace {
const int PAN_PIN = 9;
const int TILT_PIN = 10;

const int PAN_MIN = 0;
const int PAN_MAX = 180;
const int TILT_MIN = 50;
const int TILT_MAX = 130;

const int HOME_PAN = 90;
const int HOME_TILT = 90;

const unsigned long SERIAL_BAUD = 115200;

Servo panServo;
Servo tiltServo;

int currentPan = HOME_PAN;
int currentTilt = HOME_TILT;
}

int clampAngle(int value, int lo, int hi) {
  if (value < lo) return lo;
  if (value > hi) return hi;
  return value;
}

void writePose(int pan, int tilt) {
  currentPan = clampAngle(pan, PAN_MIN, PAN_MAX);
  currentTilt = clampAngle(tilt, TILT_MIN, TILT_MAX);
  panServo.write(currentPan);
  tiltServo.write(currentTilt);
}

void writeHomePose() {
  writePose(HOME_PAN, HOME_TILT);
}

void runSelfTest() {
  const int poses[][2] = {
    {HOME_PAN, HOME_TILT},
    {PAN_MIN, HOME_TILT},
    {PAN_MAX, HOME_TILT},
    {HOME_PAN, HOME_TILT},
    {HOME_PAN, TILT_MIN},
    {HOME_PAN, TILT_MAX},
    {HOME_PAN, HOME_TILT},
  };

  for (unsigned int i = 0; i < sizeof(poses) / sizeof(poses[0]); ++i) {
    writePose(poses[i][0], poses[i][1]);
    delay(700);
  }
}

bool parseAxisValue(const String& line, char axis, int* outValue) {
  int idx = line.indexOf(axis);
  if (idx < 0) {
    return false;
  }

  idx += 1;
  while (idx < line.length() && line[idx] == ' ') {
    idx += 1;
  }

  int end = idx;
  while (end < line.length() && isDigit(line[end])) {
    end += 1;
  }

  if (end <= idx) {
    return false;
  }

  *outValue = line.substring(idx, end).toInt();
  return true;
}

void handleCommand(String line) {
  line.trim();
  if (line.length() == 0) {
    return;
  }

  if (line.equalsIgnoreCase("PING")) {
    Serial.println("PONG");
    return;
  }

  int nextPan = currentPan;
  int nextTilt = currentTilt;
  bool hasPan = parseAxisValue(line, 'P', &nextPan);
  bool hasTilt = parseAxisValue(line, 'T', &nextTilt);

  if (!hasPan && !hasTilt) {
    Serial.print("ERR ");
    Serial.println(line);
    return;
  }

  writePose(nextPan, nextTilt);
  Serial.print("OK P");
  Serial.print(currentPan);
  Serial.print(" T");
  Serial.println(currentTilt);
}

void setup() {
  Serial.begin(SERIAL_BAUD);

  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);

  writeHomePose();
  delay(500);
  runSelfTest();

  Serial.print("READY P");
  Serial.print(currentPan);
  Serial.print(" T");
  Serial.println(currentTilt);
}

void loop() {
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    handleCommand(line);
  }
}
