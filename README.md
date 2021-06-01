# N-Body - Marco D'Antonio
## Testing della soluzione
La soluzione può essere eseguita ed è stata testata in un ambiente con OpenMPI il compilatore GCC, in particolare per quanto riguarda l'ambiente Docker MPI, il container dovrà essere lanciato con l'opzione `--privileged`. Il file sorgente dovrà essere compilato eseguendo:
```bash
mpicc -O3 nbody.c -o nbody.out -lm
```
e poi eseguito lanciando `mpirun`. Segue un esempio di esecuzione del programma con due processi: il primo argomento dopo il nome del programma stabilisce il numero di iterazioni della simulazione, mentre il secondo argomento il numero di particelle.
```bash
mpirun --allow-run-as-root -np 2 nbody.out 100 1000
```
È possibile definire all'interno del file sorgente una macro `DEBUG` per visualizzare alla fine dell'esecuzione il risultato della simulazione, che per semplicità stamperà a video la posizione `x` di ciascuna particella.
```c
#define DEBUG
```
## Presentazione della soluzione
Per lo sviluppo della soluzione si è operato in forma incrementale, applicando mano a mano delle ottimizzazioni alla soluzione, per poi scegliere la migliore in termini di performance. 

La soluzione finale è ora descritta sinteticamente e ad alto livello.
1. Ogni processo inizializza l'intero array di particelle.
2. Ad ogni processo è assegnata un porzione di particelle di cui calcolerà il valore nelle varie iterazioni della simulazione.
3. Mentre effettua i calcoli, ogni processo attende l'arrivo delle informazioni delle altre particelle dagli altri processi.
4. Una volta finiti i calcoli per l'iterazione corrente, ogni processo invia i suoi dati a tutti gli altri processi.
5. Prima di passare all'iterazione successiva, ogni processo attende che le informazioni delle altre particelle siano state ricevute.

## Descrizione della soluzione
Verrà ora fornita una descrizione della soluzione più dettagliata, nella quale si farà riferimento ai cinque passi della soluzioni esposti nella sua presentazione.

### Inizializzazione
Le particelle vengono inizializzate casualmente da ogni processo tuttavia, tutti i processi inizializzano il generatore pseudocasuale utilizzando lo stesso seme, questo assicura sia che i risultati siano corretti, sia la riproducibilità degli esperimenti.

```c
typedef struct body {
  float x, y, z, vx, vy, vz; 
} Body;

...

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] =  20.0f * (rand() / (float)RAND_MAX) - 10.0f;
  }
}

...

int bytes = nBodies * sizeof(Body);
Body *computationBodies = malloc(bytes);
Body *receiveBodies = malloc(bytes);

randomizeBodies((float *) computationBodies, 6 * nBodies);
```
Durante questa fase vengono allocati due array di `Body`, in quanto il primo, `computationBodies`, viene utilizzato per mantenere le particelle su cui si stanno effettuando i calcoli, mentre l'altro, `receiveBodies`, viene utilizzato per mantenere le particelle che vengono ricevute dagli altri processi, questa distinzione è necessaria in quanto la ricezione, essendo non bloccante (come si vedrà più avanti) potrebbe avvenire durante il calcolo.

### Distribuzione del carico
La porzione di array a cui ogni processo è assegnato viene calcolata in base al rango del processo all'interno del communicator, in base al numero di processi nel communicator e in base al numero di particelle da calcolare. Sono state definite tre macro che prendono in input questi valori e forniscono il numero di particelle su cui un processo si deve concentrare e l'indice di inizio e di fine della sua parte all'interno dell'array di particelle.

```c
int *receiveCounts = malloc(sizeof(int) * commSize);
int *displacements = malloc(sizeof(int) * commSize);
for (int i = 0; i < commSize; i++) {
  receiveCounts[i] = workload(i, commSize, nBodies);
  displacements[i] = startIndex(i, commSize, nBodies);
}
```
Gli array che mantengono le informazioni sul carico sono inizializzati per tutti i processi, questo perché queste informazioni sono utili a tutti nella fase di ricezione per definire dove i dati debbano essere inseriti e in che numero essi siano.

### Calcolo e ricezione
La ricezione dei dati avviene in modalità non bloccante e i dati ricevuti vengono inseriti all'interno dell'array `receiveBodies` visto in precedenza. Successivamente, vengono effettuati i calcoli per stabilire i valori della porzione di array assegnata per l'iterazione attuale.

```c
int start = startIndex(rank, commSize, nBodies);
int end = endIndex(rank, commSize, nBodies);

MPI_Request *receiveRequests = malloc(sizeof(MPI_Request) * (commSize - 1));
MPI_Request *sendRequests = malloc(sizeof(MPI_Request) * (commSize - 1));
for (int iter = 1; iter <= nIters; iter++) {

  for (int i = 0, j = 0; i < commSize; i++) 
    if (i != rank)
      MPI_Irecv(&receiveBodies[displacements[i]], receiveCounts[i], body, i, DATA_TAG, MPI_COMM_WORLD, &receiveRequests[j++]);

  for (int i = start; i < end; i++) { 
    float Fx = 0.0f;
    float Fy = 0.0f;
    float Fz = 0.0f;

    for (int j = 0; j < nBodies; j++) {
      float dx = computationBodies[j].x - computationBodies[i].x;
      float dy = computationBodies[j].y - computationBodies[i].y;
      float dz = computationBodies[j].z - computationBodies[i].z;
      float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3;
      Fy += dy * invDist3;
      Fz += dz * invDist3;
    }

    computationBodies[i].vx += dt * Fx;
    computationBodies[i].vy += dt * Fy;
    computationBodies[i].vz += dt * Fz;
  }

  for (int i = start; i < end; i++) {
    computationBodies[i].x += computationBodies[i].vx * dt;
    computationBodies[i].y += computationBodies[i].vy * dt;
    computationBodies[i].z += computationBodies[i].vz * dt;
  }

  ...

} // end for
```

### Invio dati
Dopo il calcolo dei valori per l'iterazione attuale, i dati vengono inviati a tutti gli altri processi, assicurandosi prima che tutti l'invio dell'iterazione precedente sia stato effettuato.

```c
for (int iter = 1; iter <= nIters; iter++) {

  ...

  if (iter != 1)
    MPI_Waitall(commSize - 1, sendRequests, MPI_STATUSES_IGNORE);

  for (int i = 0, j = 0; i < commSize; i++)
    if (i != rank) 
      MPI_Irsend(&computationBodies[displacements[rank]], receiveCounts[rank], body, i, DATA_TAG, MPI_COMM_WORLD, &sendRequests[j++]);    
  
  ...
  
} //end for
```
Anche l'invio dei dati è effettuato in maniera non bloccante, e in particolare si può sfruttare un invio in modalità ready, poiché la receive corrispondente è già stata inviata dagli altri processi all'inizio dell'iterazione.

### Preparazione iterazione successiva
In quest'ultimo passo vengono effettuate tre operazioni fondamentali: 
1. Il processo attende che eventuali receive vengano completate. Si noti come non necessariamente si dovrà aspettare: la receive in modalità non bloccante viene effettuata prima della fase di computazione, quindi è possibile che i risultati siano già arrivati al processo.
2. Si effettua la copia della porzione di array che il processo attuale ha calcolato dal buffer di calcolo `computationBodies` al buffer di ricezione `receiveBodies`.
3. Si effettua uno scambio degli indirizzi a cui puntano `computationBodies` e `receiveBodies`, in questo modo nella prossima iterazione `computationBodies` conterrà i dati calcolati da tutti gli altri processi (e dal processo attuale) durante l'iterazione attuale.
```c
for (int iter = 1; iter <= nIters; iter++) {

  ...    
  
  MPI_Waitall(commSize - 1, receiveRequests, MPI_STATUSES_IGNORE);

  memcpy(&receiveBodies[displacements[rank]], &computationBodies[displacements[rank]], sizeof(Body) * receiveCounts[rank]);

  swap((void **) &computationBodies, (void **) &receiveBodies);
} //end for
```

## Analisi delle prestazioni
Come detto in precedenza, lo sviluppo della soluzione è stato effettuato in maniera incrementale, questo ha permesso di comparare (seppur in locale) le performance delle varie versioni della soluzione. La soluzione descritta nella sezione precedente è quindi la migliore in termini di performance. 

Tuttavia, all'interno di questa analisi, la soluzione proposta verrà comparata anche con un'altra versione che risolve il problema. Il modo in cui quest'ultima soluzione opera è simile a quella presentata se non che, piuttosto che forzatamente aspettare alla fine di ogni iterazione l'arrivo dei dati dagli altri processi, questa procede alla prossima iterazione, mano a mano operando sui dati che arrivano dagli altri processori.

Gli esperimenti sono stati effettuati su un numero massimo di otto istanze `m4.large`, ognuna delle quali ospita nella configurazione di default un singolo CPU core, che esegue in multithreading due thread, i quali sono visibili all'applicazione come (virtual) CPU. Per questa ragione, gli esperimenti di weak e strong scaling sono stati ripetuti in due modi: prima con un mapping dei processi per nodo e poi per slot. Nella primo mapping viene lanciato un processo per nodo, ciclando tra essi, mentre nel secondo mapping vengono prima saturati i singoli nodi.
