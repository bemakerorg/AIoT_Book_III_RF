/*
  AIoT Parte 3 — Lezione 01
  "hello_world_mio": il primo modello addestrato da voi che gira su Arduino.

  Il modello, addestrato nel notebook Colab di questa lezione, calcola il seno di un numero.
  Lo sketch gli passa in continuazione i punti di un giro completo, stampa la risposta accanto
  al valore vero e usa il risultato per far respirare il LED della scheda.

  Questo file è lo SCHELETRO STANDARD che ritroverete, identico nella struttura, in tutti i
  progetti del libro:

      setup()  = preparare una volta sola:  modello -> attrezzi -> memoria -> sensore
      loop()   = ripetere per sempre:       leggi -> scrivi -> Invoke() -> leggi -> agisci

  Nei prossimi progetti cambieranno solo due cose: da dove arrivano i dati (qui: un calcolo;
  in Lezione 02: il microfono) e cosa si fa con il risultato.
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

// L'ARENA: il tavolo da lavoro dove l'interprete appoggia i risultati intermedi.
// Non dipende da quanto è grande il modello, ma dalla coppia di strati consecutivi più
// ingombrante. Per questo modello i risultati intermedi occupano appena 34 byte: 3 KB sono
// abbondanti, e servono soprattutto alle strutture interne dell'interprete.
// Lo sketch stampa all'avvio quanti byte usa davvero, così potete stringere con cognizione.
constexpr int kDimensioneArena = 3 * 1024;
alignas(16) uint8_t arena[kDimensioneArena];

constexpr float kDueGreco = 6.28318530718f;
constexpr int kPassiPerGiro = 100;
int passo = 0;

}  // namespace

void setup() {
  Serial.begin(9600);
  unsigned long inizio = millis();
  while (!Serial && (millis() - inizio) < 4000) { }

  tflite::InitializeTarget();

  // --- 1. Il modello -------------------------------------------------------
  // GetModel non copia niente: si limita a "guardare dentro" l'array di byte di model.cpp.
  modello = tflite::GetModel(g_model);
  if (modello->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Il modello e' di una versione che questa libreria non conosce.");
    return;
  }

  // --- 2. Gli attrezzi -----------------------------------------------------
  // Il nostro modello usa UNA sola operazione: FULLY_CONNECTED, cioe' i quattro strati Dense
  // che avete scritto su Colab. Chiedendo solo quella, invece della cassetta degli attrezzi
  // al completo (AllOpsResolver), il programma finito occupa molta meno memoria.
  // Il numero fra parentesi angolari e' quante operazioni diverse registriamo: qui una.
  static tflite::MicroMutableOpResolver<1> attrezzi;
  if (attrezzi.AddFullyConnected() != kTfLiteOk) {
    Serial.println("Non sono riuscito a registrare l'operazione FULLY_CONNECTED.");
    return;
  }

  // --- 3. L'interprete e la memoria ---------------------------------------
  static tflite::MicroInterpreter interprete_statico(modello, attrezzi, arena,
                                                     kDimensioneArena);
  interprete = &interprete_statico;

  if (interprete->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() fallita: l'arena e' troppo piccola, alzatela.");
    interprete = nullptr;
    return;
  }

  ingresso = interprete->input(0);
  uscita = interprete->output(0);

  Serial.print("Modello: ");
  Serial.print(g_model_len);
  Serial.print(" byte.  Arena usata davvero: ");
  Serial.print(interprete->arena_used_bytes());
  Serial.print(" byte su ");
  Serial.println(kDimensioneArena);

  // --- 4. Il "sensore" -----------------------------------------------------
  // Qui non c'e' nessun sensore: i dati ce li calcoliamo noi. Nella Lezione 02, al suo posto,
  // ci sara' l'accensione del microfono.
  pinMode(LED_BUILTIN, OUTPUT);

  Serial.println();
  Serial.println("        x       previsto        vero");
}

void loop() {
  if (interprete == nullptr) return;  // qualcosa e' andato storto in setup()

  // --- Leggi: da dove arriva il dato --------------------------------------
  float x = kDueGreco * passo / kPassiPerGiro;

  // --- Scrivi nell'ingresso ------------------------------------------------
  // Il modello ragiona in numeri interi da -128 a +127. Per passargli il nostro x lo
  // traduciamo con i due numeri che il convertitore ha calcolato su Colab e ha lasciato
  // scritti dentro il modello: la scala e lo zero. E' la stessa formula che avete visto
  // nell'ultima cella di controllo del notebook.
  int intero = (int)(roundf(x / ingresso->params.scale) + ingresso->params.zero_point);
  if (intero < -128) intero = -128;
  if (intero > 127) intero = 127;
  ingresso->data.int8[0] = (int8_t)intero;

  // --- Invoke: il modello lavora -------------------------------------------
  if (interprete->Invoke() != kTfLiteOk) {
    Serial.println("Invoke() fallita.");
    return;
  }

  // --- Leggi l'uscita ------------------------------------------------------
  // La traduzione inversa: da numero intero a numero con la virgola.
  float y = (uscita->data.int8[0] - uscita->params.zero_point) * uscita->params.scale;

  // --- Agisci --------------------------------------------------------------
  // Serial.print stampa i numeri in modo leggibile. La libreria mette a disposizione anche
  // una sua funzione, MicroPrintf, ma quella scrive i numeri con la virgola in un formato
  // illeggibile (per esempio "1.3194684*2^2" invece di "5.2779"): meglio evitarla.
  Serial.print("  ");
  Serial.print(x, 4);
  Serial.print("\t");
  Serial.print(y, 4);
  Serial.print("\t");
  Serial.println(sinf(x), 4);

  // Il seno va da -1 a +1; la luminosita' del LED da 0 a 255.
  int luminosita = (int)(127.5f * (y + 1.0f));
  luminosita = constrain(luminosita, 0, 255);
  analogWrite(LED_BUILTIN, luminosita);

  passo = (passo + 1) % kPassiPerGiro;
  delay(60);
}
