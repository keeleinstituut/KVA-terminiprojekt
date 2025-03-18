# KVA-terminiprojekt

### Kogu rakenduse üles seadmine *docker-compose*'iga

Täida **.env** fail.

Konfigureerida **docker-compose.yml** failis: 
* Postgre andmete asukoht: `{pg_andmestike_failitee}:/var/lib/postgresql/data`

* Qdranti andmete asukoht: `{qdranti_andmestike_failitee}:/qdrant/storage`

* Autentimiseks kasutajainfo asukoht: `{credentials_fail/credentials.json}:/app/config/credentials.json`, näiteks paroolita kasutajakonto jaoks:
*{"kulaline": ""}*

~~~
docker compose up
~~~

#### Kui on vaja muuta konfiguratsiooni:
Muuta vajadusel konfiguratsiooni **/config/config.json** -- viimasel juhul tuleb teha ka kva-terminiprojekti image build, nt docker compose failis   
~~~
 terms:
    build: ./
~~~


### Rakenduse osade üles seadmine:

1) Andmebaaside üles seadmine

Eemaldada docker-compose failist `terms` sektsioon.

Konfigureerida **docker-compose.yml** failis: 
* Postgre andmete asukoht: `{pg_andmestike_failitee}:/var/lib/postgresql/data`

* Qdranti andmete asukoht: `{qdranti_andmestike_failitee}:/qdrant/storage`
 
~~~
docker compose up
~~~

2) Kasutajaliidese üles seadmine

Täida .env fail.
Muuta vajadusel konfiguratsiooni **/config/config.json**

~~~
docker build --progress=plain -t kva-ui .
docker run --env-file .env --name kva-ui -v {credentials_fail/credentials.json}:/app/config/credentials.json -p 5006:5006 -it kva-ui
~~~


### Mudelid

Rakendust saab kasutada OpenAI ja Anthropicu Claude'i mudelitega. Mudeli vahetamiseks vahetada mudeli nimi failis **config.json**:
    ```
 "llm": {
        "model": "gpt-4o"
    },
    ```