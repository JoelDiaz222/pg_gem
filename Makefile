MODULE_big = pg_gembed
OBJS = src/pg_gembed.o src/embedding_worker.o

EXTENSION = pg_gembed
EXTVERSION = 0.1.0
DATA = sql/$(EXTENSION)--$(EXTVERSION).sql

PG_CONFIG = pg_config

PG_CPPFLAGS = \
	-I/opt/homebrew/include/postgresql@18/server \
	-I/opt/homebrew/Cellar/pgvector/0.8.1/include/postgresql@18/server/extension/vector

RUST_DIR = pg_gem_core
RUST_TARGET = $(RUST_DIR)/target/release
RUST_LIB = $(RUST_TARGET)/libpg_gem_core.dylib

SHLIB_LINK = \
	-L$(RUST_TARGET) \
	-lpg_gem_core \
	-undefined dynamic_lookup

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

$(MODULE_big).dylib: $(OBJS) $(RUST_LIB)

$(RUST_LIB):
	cd $(RUST_DIR) && cargo build --release

clean:
	rm -f $(OBJS) $(MODULE_big).dylib
	cd $(RUST_DIR) && cargo clean
