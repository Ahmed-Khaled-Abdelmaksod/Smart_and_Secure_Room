#include <TinyDHT.h>

// --- DHT Sensor Setup ---
#define DHTPIN 2       // DHT data pin
#define DHTTYPE DHT11  // DHT11 or DHT22
DHT dht(DHTPIN, DHTTYPE);

// --- Other Sensors ---
#define PIR_PIN 4        // PIR sensor output pin
#define SMOKE_PIN A0     // Smoke sensor analog pin
#define LED_PIN 13       // Built-in LED pin

// --- Ultrasonic Sensor ---
#define TRIG_PIN 7
#define ECHO_PIN 6

// --- Thresholds ---
#define TEMP_THRESHOLD 35.0
#define SMOKE_THRESHOLD 300   // Adjust based on your MQ sensor readings

void setup() {
  Serial.begin(9600);
  dht.begin();

  pinMode(LED_PIN, OUTPUT);
  pinMode(PIR_PIN, INPUT);
  pinMode(SMOKE_PIN, INPUT);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
}

float getDistanceCM() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000); // Timeout 30ms
  float distance = (duration * 0.0343) / 2; // Convert to cm
  if (duration == 0) distance = -1; // No reading
  return distance;
}

void loop() {
  // --- Read DHT values ---
  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();

  if (isnan(humidity) || isnan(temperature)) {
    Serial.println("{\"error\": \"Failed to read DHT sensor\"}");
    delay(1000);
    return;
  }

  // --- Read PIR sensor ---
  int motionDetected = digitalRead(PIR_PIN);

  // --- Read Smoke sensor ---
  int smokeValue = analogRead(SMOKE_PIN);

  // --- Read Ultrasonic sensor ---
  float distanceCM = getDistanceCM();

  // --- LED Control Logic ---
  if (temperature > TEMP_THRESHOLD || motionDetected == HIGH || smokeValue > SMOKE_THRESHOLD) {
    digitalWrite(LED_PIN, HIGH);
  } else {
    digitalWrite(LED_PIN, LOW);
  }

  // --- Send Data to Raspberry Pi (JSON format) ---
  Serial.print("{\"temperature\": ");
  Serial.print(temperature);
  Serial.print(", \"humidity\": ");
  Serial.print(humidity);
  Serial.print(", \"motion\": ");
  Serial.print(motionDetected);
  Serial.print(", \"smoke\": ");
  Serial.print(smokeValue);
  Serial.print(", \"distance\": ");
  Serial.print(distanceCM);
  Serial.println("}");

  delay(1000);
}
