# Creating a Vector Database with PostgreSQL and pgvector

## Steps to Create `langchain_vectordb`
```

### 4. Enable pgvector Extension
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 5. Create a Vector Table
```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536)
);
```

### 6. Create an Index for Similarity Search
```sql
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### 7. Verify the Database
```sql
\d
\d documents
```

### 8. Exit PostgreSQL
```sql
\q
```

## Usage Example in Python

```python
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings.openai import OpenAIEmbeddings

CONNECTION_STRING = "postgresql://postgres:password@localhost:5432/langchain_vectordb"

embeddings = OpenAIEmbeddings()
vector_store = PGVector.from_documents(
    documents=docs,
    embedding=embeddings,
    connection_string=CONNECTION_STRING,
    table_name="documents"
)
```
