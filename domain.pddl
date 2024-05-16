(define (domain graph-test)

(:requirements :strips :typing :equality :negative-preconditions)

(:types
	type0 type1 - node
	)

(:predicates
	(link ?n0 - node ?n1 - node)
	)

(:action transformation0
	:parameters (?n1t0 - type0 ?n10t1 - type1 ?n19t1 - type1 ?n15t1 - type1 	)
	:precondition (and
		(not (= ?n10t1 ?n19t1))
		(not (= ?n10t1 ?n15t1))
		(not (= ?n19t1 ?n15t1))
	)
	:effect (and
		(link ?n1t0 ?n10t1)
		(link ?n10t1 ?n1t0)
		(link ?n1t0 ?n19t1)
		(link ?n19t1 ?n1t0)
		(link ?n10t1 ?n15t1)
		(link ?n15t1 ?n10t1)
		(link ?n19t1 ?n15t1)
		(link ?n15t1 ?n19t1)
	)
)

)