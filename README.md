# KVA-terminiprojekt

### 1) Andmebaaside üles seadmine

Konfigureerida **docker-compose.yml** failis: 
* Postgre andmete asukoht: `{pg_andmestike_failitee}:/var/lib/postgresql/data`

* Qdranti andmete asukoht: `{qdranti_andmestike_failitee}:/qdrant/storage`
 
~~~
docker compose up
~~~

### 2) Kasutajaliidese üles seadmine

Täida .env fail.
Muuta vajadusel konfiguratsiooni **/config/config.json**

~~~
docker build --progress=plain -t kva-ui .
docker run --env-file .env --name kva-ui --shm-size=2gb -p 5006:5006 -it kva-ui
~~~
