explain analyze (SELECT generate_embeddings(
  ARRAY(SELECT 'sentence ' || g FROM generate_series(1,2500) g)
));

explain analyze (SELECT generate_embeddings_with_ids_c(
  ARRAY[1],
  ARRAY['a']
));

explain analyze (SELECT generate_embeddings_with_ids_c(
  ARRAY(SELECT i FROM generate_series(1,2500) AS i),
  ARRAY(SELECT 'sentence ' || i FROM generate_series(1,2500) AS i)
));

explain analyze (SELECT generate_embeddings_with_ids(
  ARRAY(SELECT i FROM generate_series(1,2500) AS i),
  ARRAY(SELECT 'sentence ' || i FROM generate_series(1,2500) AS i)
));
