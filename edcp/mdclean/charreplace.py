class BookClean:
    REMOVE_WORDS = None
    REPLACEMENTS = [
        (r"弓起", "引起"),
        (r"引\|", "引"),
        (r"泉液", "尿液"),
        (r"十扰", "干扰"),
        (r"(!|！){2,}", ""),
    ]
