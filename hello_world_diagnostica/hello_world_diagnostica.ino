/*
  AIoT Parte 3 — sketch di DIAGNOSTICA per hello_world
  (non è materiale del libro: serve a capire dove si rompe la catena)

  Fa tre cose:
   1. Stampa quello che la scheda vede davvero del modello: dimensioni, tipo dei numeri,
      fattore di scala e punto zero. Li stampa come NUMERI INTERI, perché la funzione
      MicroPrintf della libreria stampa i numeri con la virgola in un formato illeggibile
      (li scrive come "1.3194684*2^2" invece che come "5.28").
   2. Esegue sei prove con valori di ingresso di cui conosciamo già la risposta esatta,
      calcolata al computer sullo stesso identico file model.cpp, e dice PASSA o FALLISCE.
   3. Se le prove passano, stampa la sinusoide con Serial.print(), che scrive numeri normali.

  Aprire il Monitor Seriale a 9600 baud.
*/

#include <TensorFlowLite.h>

#include <cmath>

#include "model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

const tflite::Model* modello = nullptr;
tflite::MicroInterpreter* interprete = nullptr;
TfLiteTensor* ingresso = nullptr;
TfLiteTensor* uscita = nullptr;

constexpr int kDimensioneArena = 4 * 1024;
alignas(16) uint8_t arena[kDimensioneArena];

constexpr float kDueGreco = 6.28318530718f;
constexpr int kPassiPerGiro = 100;
int passo = 0;
bool tutto_ok = false;

// Valori calcolati al computer sullo STESSO model.cpp che sta su questa scheda.
// Se la scheda produce numeri diversi, il problema è nell'esecuzione, non nel modello.
const int kNumProve = 6;
const int kProvaPasso[kNumProve]      = {   0,  16,  25,  50,  75,  84 };
const int kProvaIngresso[kNumProve]   = {-128, -87, -64,   0,  64,  87 };
const int kProvaUscita[kNumProve]     = {  17, 107, 121,   2,-123, -94 };

// Attesi anche questi (in miliardesimi, per stamparli come interi):
const long kScalaIngressoAttesa = 24575612L;   // 0,024575612
const int  kZeroIngressoAtteso  = -128;
const long kScalaUscitaAttesa   = 7824853L;    // 0,007824853
const int  kZeroUscitaAtteso    = -2;

void stampaTensore(const char* etichetta, TfLiteTensor* t) {
  Serial.print(etichetta);
  Serial.print(" tipo=");
  Serial.print((int)t->type);
  Serial.print(" (9 = int8)  dimensioni=[");
  for (int i = 0; i < t->dims->size; i++) {
    if (i) Serial.print(",");
    Serial.print(t->dims->data[i]);
  }
  Serial.print("]  scala(miliardesimi)=");
  Serial.print((long)(t->params.scale * 1e9f));
  Serial.print("  zero=");
  Serial.println(t->params.zero_point);
}

}  // namespace

void setup() {
  Serial.begin(9600);
  unsigned long inizio = millis();
  while (!Serial && (millis() - inizio) < 4000) { }

  Serial.println();
  Serial.println(F("=== DIAGNOSTICA hello_world ==="));

  tflite::InitializeTarget();

  Serial.print(F("Lunghezza del modello dichiarata in model.cpp: "));
  Serial.println(g_model_len);

  modello = tflite::GetModel(g_model);
  Serial.print(F("Versione dello schema letta nel modello: "));
  Serial.print(modello->version());
  Serial.print(F("  (attesa "));
  Serial.print(TFLITE_SCHEMA_VERSION);
  Serial.println(F(")"));
  if (modello->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println(F("FERMO: versione dello schema sbagliata."));
    return;
  }

  static tflite::MicroMutableOpResolver<1> attrezzi;
  if (attrezzi.AddFullyConnected() != kTfLiteOk) {
    Serial.println(F("FERMO: non sono riuscito a registrare FULLY_CONNECTED."));
    return;
  }

  static tflite::MicroInterpreter interprete_statico(modello, attrezzi, arena,
                                                     kDimensioneArena);
  interprete = &interprete_statico;

  if (interprete->AllocateTensors() != kTfLiteOk) {
    Serial.println(F("FERMO: AllocateTensors() fallita, arena troppo piccola."));
    return;
  }

  Serial.print(F("Arena usata davvero: "));
  Serial.print(interprete->arena_used_bytes());
  Serial.print(F(" byte su "));
  Serial.println(kDimensioneArena);

  ingresso = interprete->input(0);
  uscita = interprete->output(0);
  stampaTensore("Ingresso:", ingresso);
  stampaTensore("Uscita  :", uscita);

  Serial.println();
  Serial.println(F("Confronto con i valori attesi (calcolati al computer):"));
  bool meta_ok = true;
  if (labs((long)(ingresso->params.scale * 1e9f) - kScalaIngressoAttesa) > 100L) {
    Serial.print(F("  DIVERSO: scala ingresso, attesa ")); Serial.println(kScalaIngressoAttesa);
    meta_ok = false;
  }
  if (ingresso->params.zero_point != kZeroIngressoAtteso) {
    Serial.print(F("  DIVERSO: zero ingresso, atteso ")); Serial.println(kZeroIngressoAtteso);
    meta_ok = false;
  }
  if (labs((long)(uscita->params.scale * 1e9f) - kScalaUscitaAttesa) > 100L) {
    Serial.print(F("  DIVERSO: scala uscita, attesa ")); Serial.println(kScalaUscitaAttesa);
    meta_ok = false;
  }
  if (uscita->params.zero_point != kZeroUscitaAtteso) {
    Serial.print(F("  DIVERSO: zero uscita, atteso ")); Serial.println(kZeroUscitaAtteso);
    meta_ok = false;
  }
  Serial.println(meta_ok ? F("  I dati del modello combaciano.")
                         : F("  I dati del modello NON combaciano."));

  Serial.println();
  Serial.println(F("Prove con valori noti:"));
  Serial.println(F(" passo  q_in atteso  q_in reale   q_out atteso  q_out reale   esito"));
  int passate = 0;
  for (int p = 0; p < kNumProve; p++) {
    float x = kDueGreco * kProvaPasso[p] / kPassiPerGiro;
    int q_in = (int)(roundf(x / ingresso->params.scale) + ingresso->params.zero_point);
    if (q_in < -128) q_in = -128;
    if (q_in > 127) q_in = 127;
    ingresso->data.int8[0] = (int8_t)q_in;

    if (interprete->Invoke() != kTfLiteOk) {
      Serial.println(F("  Invoke() fallita."));
      break;
    }
    int q_out = (int)uscita->data.int8[0];
    bool ok = (q_in == kProvaIngresso[p]) && (abs(q_out - kProvaUscita[p]) <= 2);
    if (ok) passate++;

    Serial.print(F("   "));   Serial.print(kProvaPasso[p]);
    Serial.print(F("\t   "));  Serial.print(kProvaIngresso[p]);
    Serial.print(F("\t       "));  Serial.print(q_in);
    Serial.print(F("\t          "));  Serial.print(kProvaUscita[p]);
    Serial.print(F("\t      "));  Serial.print(q_out);
    Serial.print(F("\t   "));  Serial.println(ok ? F("PASSA") : F("FALLISCE"));
  }

  Serial.println();
  Serial.print(F("Prove passate: "));
  Serial.print(passate);
  Serial.print(F(" su "));
  Serial.println(kNumProve);
  tutto_ok = (passate == kNumProve);
  Serial.println(tutto_ok
      ? F("TUTTO A POSTO: la scheda esegue il modello correttamente.")
      : F("PROBLEMA: la scheda produce numeri diversi dal computer."));

  Serial.println();
  Serial.println(F("Ora la sinusoide, stampata con Serial.print (numeri leggibili):"));
  Serial.println(F("        x        previsto      vero"));
  delay(2000);
}

void loop() {
  if (interprete == nullptr || ingresso == nullptr) return;

  float x = kDueGreco * passo / kPassiPerGiro;

  int q_in = (int)(roundf(x / ingresso->params.scale) + ingresso->params.zero_point);
  if (q_in < -128) q_in = -128;
  if (q_in > 127) q_in = 127;
  ingresso->data.int8[0] = (int8_t)q_in;

  if (interprete->Invoke() != kTfLiteOk) return;

  float y = (uscita->data.int8[0] - uscita->params.zero_point) * uscita->params.scale;

  Serial.print("  ");
  Serial.print(x, 4);
  Serial.print("\t");
  Serial.print(y, 4);
  Serial.print("\t");
  Serial.println(sinf(x), 4);

  int luminosita = (int)(127.5f * (y + 1.0f));
  luminosita = constrain(luminosita, 0, 255);
  analogWrite(LED_BUILTIN, luminosita);

  passo = (passo + 1) % kPassiPerGiro;
  delay(60);
}
