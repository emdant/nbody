In locale, esegui:
```
./launch-instances.sh
```

Ora sei sul master, esegui:
```
source ./setup.sh
```

Aspetta che termini l'installazione di MPI su ogni nodo, poi esegui:
```
sudo cp ./afterSetup.sh /home/pcpc
sudo login pcpc
#password: root
source ./afterSetup.sh
```

Torna in locale
```
./terminate-instances.sh
```