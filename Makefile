MODULE_big = pg_gembed
OBJS = src/pg_gembed.o src/embedding_worker.o

EXTENSION = pg_gembed
EXTVERSION = 0.1.0
DATA = sql/$(EXTENSION)--$(EXTVERSION).sql

PG_CONFIG = pg_config

PG_INCLUDEDIR = $(shell $(PG_CONFIG) --includedir-server)

VECTOR_INC_DIR ?= $(PG_INCLUDEDIR)/extension/vector

PG_CPPFLAGS = -I$(VECTOR_INC_DIR)

GEMBED_DIR = gembed
GEMBED_TARGET = $(GEMBED_DIR)/target/release
GEMBED_LIB = $(GEMBED_TARGET)/libgembed.a

SHLIB_LINK = \
	-L$(GEMBED_TARGET) \
	-lgembed

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	SHLIB_LINK += -undefined dynamic_lookup
endif

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

$(shlib): $(GEMBED_LIB)

$(GEMBED_LIB):
	cd $(GEMBED_DIR) && cargo build --release

clean:
	rm -f $(OBJS) $(MODULE_big).so $(MODULE_big).dylib
	cd $(GEMBED_DIR) && cargo clean
