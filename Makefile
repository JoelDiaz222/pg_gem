MODULE_big = pg_gembed
OBJS = src/pg_gembed.o src/embedding_worker.o

EXTENSION = pg_gembed
EXTVERSION = 0.1.0
DATA = sql/$(EXTENSION)--$(EXTVERSION).sql

PG_CONFIG = pg_config

PG_CPPFLAGS = \
	-I/opt/homebrew/include/postgresql@18/server \
	-I/opt/homebrew/Cellar/pgvector/0.8.1/include/postgresql@18/server/extension/vector

GEMBED_DIR = gembed
GEMBED_TARGET = $(GEMBED_DIR)/target/release
GEMBED_LIB = $(GEMBED_TARGET)/libgembed.dylib

SHLIB_LINK = \
	-L$(GEMBED_TARGET) \
	-lgembed \
	-undefined dynamic_lookup

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

$(MODULE_big).dylib: $(OBJS) $(GEMBED_LIB)

$(GEMBED_LIB):
	cd $(GEMBED_DIR) && cargo build --release

clean:
	rm -f $(OBJS) $(MODULE_big).dylib
	cd $(GEMBED_DIR) && cargo clean
