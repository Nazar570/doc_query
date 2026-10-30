# Doc Query

Doc Query to aplikacja do zadawania pytań o własne dokumenty. Wgraj plik, zadaj pytanie i otrzymaj odpowiedź z odwołaniami do fragmentów, na których została oparta.

Obsługiwane formaty – PDF z warstwą tekstową, TXT i Markdown. Aplikacja udostępnia interfejs w przeglądarce oraz API.

## Jak to działa

```text
Dokument → podział na fragmenty → embeddingi → Pinecone
                                                  ↓
Pytanie → embedding → wyszukiwanie w Pinecone → fragmenty → Gemini → odpowiedź ze źródłami
```

Tekst dokumentu jest dzielony na fragmenty i zamieniany na wektory przez model multilingual-e5-small. Pinecone przechowuje wektory i wyszukuje fragmenty podobne do pytania. Gemini przygotowuje odpowiedź na podstawie znalezionych fragmentów. Jeśli nie ma w nich wystarczających informacji, aplikacja zwraca odpowiedni komunikat.

## Uruchomienie

Sklonuj repozytorium i zainstaluj zależności:

```bash
git clone https://github.com/Nazar570/doc_query.git
cd doc_query
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Uzupełnij plik `.env`:

```dotenv
PINECONE_API_KEY=twój_klucz
PINECONE_INDEX_HOST=adres_indeksu
PINECONE_NAMESPACE=doc_query
GEMINI_API_KEY=twój_klucz
LLM_MODEL=nazwa_modelu
```

Indeks Pinecone musi mieć wymiar **384**. Model embeddingów jest używany lokalnie – przed pierwszym uruchomieniem pobierz go do pamięci podręcznej:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"
uvicorn app.main:app --reload
```

Otwórz aplikację pod adresem [localhost:8000](http://127.0.0.1:8000). Interaktywna dokumentacja API znajduje się pod [localhost:8000/docs](http://127.0.0.1:8000/docs).

## Przykład użycia

Po uruchomieniu aplikacji otwórz stronę w przeglądarce i dodaj dokument. Gdy jego status zmieni się na `ready`, możesz zadać pytanie. Odpowiedź zawiera odwołania do wykorzystanych fragmentów.

Dokument można również dodać przez API:

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -F "file=@test_docs/onboarding.txt"
```

Następnie wyślij pytanie:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Co opisuje dokument?","k":4}'
```

Opcjonalne `source_id` w treści zapytania ogranicza wyszukiwanie do wybranego dokumentu. Identyfikator otrzymasz po dodaniu pliku lub z listy dokumentów.

## Dokumenty

Aplikacja przyjmuje do 10 dokumentów, każdy o rozmiarze do 1 MiB. Pliki TXT i Markdown muszą być zapisane w UTF-8. PDF może mieć maksymalnie 50 stron i musi zawierać tekst możliwy do odczytania – skany bez warstwy tekstowej nie są obsługiwane. W katalogu `test_docs` znajdują się przykładowe pliki.

## API

- `GET /health` – stan serwera
- `POST /documents` – dodanie dokumentu
- `GET /documents` – lista dokumentów i ich statusów
- `GET /documents/{source_id}/chunks` – fragmenty dokumentu
- `POST /documents/{source_id}/refresh` – odświeżenie statusu
- `POST /search` – wyszukiwanie fragmentów
- `POST /ask` – odpowiedź na pytanie wraz ze źródłami

W `/search` i `/ask` można podać `source_id`, aby ograniczyć wyszukiwanie do jednego dokumentu.

## Technologie

Python i FastAPI – API i aplikacja webowa  
Sentence Transformers – embeddingi tekstu  
Pinecone – wyszukiwanie wektorowe  
Gemini – generowanie odpowiedzi  
pypdf – odczyt plików PDF
