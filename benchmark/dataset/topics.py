"""
Source data for the semantic-retrieval benchmark dataset.

Deliberately plain data, not a generator with any randomness — every run of
build_dataset.py against this exact file produces byte-identical output,
which is the whole point of calling the dataset "deterministic": a benchmark
whose ground truth could shift between runs would make Recall/MRR numbers
meaningless to compare over time.

Each topic contributes a handful of "memory" sentences (what a personal
assistant might have actually stored) and a handful of queries against them,
each query tagged with the retrieval-difficulty category it's meant to probe:

  exact        - shares the key content words with the memory verbatim.
  morphology   - same root words, different grammatical form/case/number -
                 a token-exact lexical matcher (see lexical.py, which mirrors
                 Mobile_mem0's own MemoryRanking.tokenize/overlap) treats
                 "мясо" and "мясом" as unrelated tokens; embeddings shouldn't
                 care.
  synonym      - key words replaced with a true synonym, not a paraphrase of
                 the whole sentence.
  paraphrase   - the same fact restated with different words and structure.
  low_overlap  - same meaning, almost no shared vocabulary at all - the
                 category semantic retrieval exists for.
  identifier   - an exact code, name, or number where a lexical/exact-string
                 match is plausibly *better* than an embedding, which tends
                 to blur short alphanumeric tokens.

"unrelated negatives" are not per-topic — see NEGATIVE_QUERIES at the bottom:
queries with no ground truth in this dataset at all, used to measure how
confidently (or not) retrieval scores something that isn't actually here.

To extend: append a new dict to TOPICS (or a new string to NEGATIVE_QUERIES)
and rerun `python build_dataset.py` — ids are assigned deterministically from
each topic's position and each memory/query's index within it, so appending
at the end never renumbers anything that already exists.
"""

TOPICS = [
    {
        "id": "diet",
        "memories": [
            "Пользователь придерживается вегетарианской диеты и не ест мясо.",
            "У пользователя аллергия на арахис и любые орехи.",
            "Пользователь готовит на оливковом масле, а не на подсолнечном.",
            "По утрам пользователь пьёт зелёный чай вместо кофе.",
            "Пользователь избегает глютена из-за непереносимости.",
        ],
        "queries": [
            {"text": "Ест ли пользователь мясо?", "category": "exact", "memory_indices": [0]},
            {"text": "Можно ли ему есть блюдо с орехами?", "category": "morphology", "memory_indices": [1]},
            {"text": "Он употребляет мясные продукты?", "category": "synonym", "memory_indices": [0]},
            {"text": "Расскажи, какого рациона питания придерживается пользователь.", "category": "paraphrase", "memory_indices": [0]},
            {"text": "Что ему стоит выбрать в стейк-хаусе?", "category": "low_overlap", "memory_indices": [0]},
        ],
    },
    {
        "id": "travel",
        "memories": [
            "Пользователь забронировал номер в отеле, код брони RZ-88214.",
            "Пользователь предпочитает путешествовать поездом, а не самолётом.",
            "У пользователя виза в Японию действует до 2027 года.",
            "Пользователь любит останавливаться в маленьких гостевых домах, а не в сетевых отелях.",
            "Рейс пользователя в Стамбул — SU1234, вылет утром.",
        ],
        "queries": [
            {"text": "Какой у меня код брони отеля RZ-88214?", "category": "identifier", "memory_indices": [0]},
            {"text": "Каким рейсом полетит пользователь в Стамбул?", "category": "morphology", "memory_indices": [4]},
            {"text": "Он летает самолётами или ездит поездами?", "category": "synonym", "memory_indices": [1]},
            {"text": "До какого года действительна виза пользователя в Японию?", "category": "paraphrase", "memory_indices": [2]},
            {"text": "Понравится ли ему сетевой отель с конвейерным сервисом?", "category": "low_overlap", "memory_indices": [3]},
        ],
    },
    {
        "id": "work",
        "memories": [
            "Пользователь работает бэкенд-разработчиком в стартапе.",
            "Проект пользователя использует Kotlin и PostgreSQL.",
            "У пользователя дедлайн по релизу 15 октября.",
            "Пользователь предпочитает удалённую работу офису.",
            "Тимлид пользователя зовёт Марина.",
        ],
        "queries": [
            {"text": "Кем работает пользователь?", "category": "exact", "memory_indices": [0]},
            {"text": "На каком языке написан его проект?", "category": "morphology", "memory_indices": [1]},
            {"text": "Он трудится из дома или ходит в офис?", "category": "synonym", "memory_indices": [3]},
            {"text": "Когда у него горит срок сдачи релиза?", "category": "paraphrase", "memory_indices": [2]},
            {"text": "Кому он отчитывается по работе?", "category": "low_overlap", "memory_indices": [4]},
        ],
    },
    {
        "id": "family",
        "memories": [
            "У пользователя есть младшая сестра по имени Аня.",
            "Родители пользователя живут в Казани.",
            "У пользователя двое детей: сын и дочь.",
            "Жена пользователя работает врачом-педиатром.",
            "Пользователь навещает родителей на майские праздники.",
        ],
        "queries": [
            {"text": "В каком городе живут его родители?", "category": "morphology", "memory_indices": [1]},
            {"text": "Сколько у него ребят — сыновей и дочерей?", "category": "synonym", "memory_indices": [2]},
            {"text": "Кем по профессии работает супруга пользователя?", "category": "paraphrase", "memory_indices": [3]},
            {"text": "В какое время года он ездит к родителям в гости?", "category": "low_overlap", "memory_indices": [4]},
        ],
    },
    {
        "id": "pets",
        "memories": [
            "У пользователя кот породы мейн-кун по кличке Барон.",
            "Кот пользователя не переносит рыбный корм.",
            "Пользователь водит кота к ветеринару раз в полгода.",
            "У кота пользователя есть отдельная переноска для перелётов.",
        ],
        "queries": [
            {"text": "Как зовут кота пользователя?", "category": "exact", "memory_indices": [0]},
            {"text": "Какой корм плохо переносится котом?", "category": "morphology", "memory_indices": [1]},
            {"text": "Он часто возит питомца к врачу-животному?", "category": "synonym", "memory_indices": [2]},
            {"text": "Готов ли пользователь взять кота с собой в поездку?", "category": "low_overlap", "memory_indices": [3]},
        ],
    },
    {
        "id": "health",
        "memories": [
            "У пользователя аллергия на пыльцу берёзы весной.",
            "Пользователь принимает витамин D по назначению врача.",
            "Пользователь занимается бегом три раза в неделю.",
            "У пользователя было сотрясение мозга в 2019 году.",
            "Пользователь спит по 6 часов в сутки на будних днях.",
        ],
        "queries": [
            {"text": "Какие витамины ему прописали?", "category": "morphology", "memory_indices": [1]},
            {"text": "Сколько раз в неделю он тренируется бегом?", "category": "synonym", "memory_indices": [2]},
            {"text": "Расскажи о травме головы, которая была у пользователя.", "category": "paraphrase", "memory_indices": [3]},
            {"text": "Хватает ли ему отдыха по будням?", "category": "low_overlap", "memory_indices": [4]},
        ],
    },
    {
        "id": "tech",
        "memories": [
            "У пользователя телефон Pixel 9 Pro.",
            "Пользователь пользуется VPN для рабочих задач.",
            "Ноутбук пользователя работает на Linux.",
            "Пользователь хранит пароли в менеджере паролей Bitwarden.",
            "У пользователя механическая клавиатура с переключателями Cherry MX.",
        ],
        "queries": [
            {"text": "Какой операционной системой пользуется его ноутбук?", "category": "morphology", "memory_indices": [2]},
            {"text": "Где он держит свои пароли?", "category": "synonym", "memory_indices": [3]},
            {"text": "Использует ли он защищённое соединение для работы?", "category": "paraphrase", "memory_indices": [1]},
            {"text": "Любит ли он тактильный отклик при печати?", "category": "low_overlap", "memory_indices": [4]},
        ],
    },
    {
        "id": "wifi",
        "memories": [
            "Пароль от домашнего Wi-Fi пользователя: Sunflower-92X.",
            "Роутер пользователя стоит в коридоре у входной двери.",
            "Пользователь перезагружает роутер, если интернет пропадает.",
        ],
        "queries": [
            {"text": "Какой пароль от Wi-Fi Sunflower-92X?", "category": "identifier", "memory_indices": [0]},
            {"text": "Где расположен роутер в квартире?", "category": "morphology", "memory_indices": [1]},
            {"text": "Что он делает, когда пропадает связь?", "category": "paraphrase", "memory_indices": [2]},
        ],
    },
    {
        "id": "finance",
        "memories": [
            "Номер полиса страхования пользователя: INS-774521.",
            "Пользователь копит на первоначальный взнос по ипотеке.",
            "Пользователь платит за подписку на стриминг раз в месяц.",
            "У пользователя есть отдельный счёт для отпускных накоплений.",
        ],
        "queries": [
            {"text": "Какой номер страхового полиса INS-774521?", "category": "identifier", "memory_indices": [0]},
            {"text": "На что он откладывает деньги для покупки жилья?", "category": "synonym", "memory_indices": [1]},
            {"text": "Как часто списывается плата за подписку?", "category": "morphology", "memory_indices": [2]},
            {"text": "Есть ли у него сбережения специально для путешествий?", "category": "low_overlap", "memory_indices": [3]},
        ],
    },
    {
        "id": "car",
        "memories": [
            "У пользователя автомобиль Toyota Corolla, номер А123ВС777.",
            "Пользователь проходит техосмотр каждую весну.",
            "Пользователь паркуется на подземной парковке у работы.",
        ],
        "queries": [
            {"text": "Какой госномер А123ВС777 у машины пользователя?", "category": "identifier", "memory_indices": [0]},
            {"text": "В какое время года он проходит техобслуживание автомобиля?", "category": "paraphrase", "memory_indices": [1]},
            {"text": "Где он оставляет машину рядом с офисом?", "category": "synonym", "memory_indices": [2]},
        ],
    },
    {
        "id": "hobby_music",
        "memories": [
            "Пользователь любит слушать джаз по вечерам.",
            "Пользователь учится играть на гитаре второй год.",
            "Любимая группа пользователя — Radiohead.",
        ],
        "queries": [
            {"text": "Какую музыку любит слушать пользователь вечером?", "category": "exact", "memory_indices": [0]},
            {"text": "На каком инструменте он учится играть?", "category": "morphology", "memory_indices": [1]},
            {"text": "Какая у него любимая музыкальная группа?", "category": "paraphrase", "memory_indices": [2]},
            {"text": "Стоит ли включать ему на вечеринке тяжёлый метал?", "category": "low_overlap", "memory_indices": [0]},
        ],
    },
    {
        "id": "hobby_books",
        "memories": [
            "Пользователь сейчас читает роман Достоевского «Идиот».",
            "Пользователь предпочитает бумажные книги электронным.",
            "Пользователь ведёт список прочитанных книг за год.",
        ],
        "queries": [
            {"text": "Какую книгу сейчас читает пользователь?", "category": "exact", "memory_indices": [0]},
            {"text": "Читает ли он книги с экрана или в печатном виде?", "category": "synonym", "memory_indices": [1]},
            {"text": "Следит ли он за тем, сколько книг прочитал?", "category": "paraphrase", "memory_indices": [2]},
        ],
    },
    {
        "id": "language",
        "memories": [
            "Пользователь учит испанский язык на курсах два раза в неделю.",
            "Цель пользователя — сдать экзамен DELE B2 в следующем году.",
            "Пользователь смотрит фильмы на испанском с субтитрами для практики.",
        ],
        "queries": [
            {"text": "Какой язык учит пользователь?", "category": "exact", "memory_indices": [0]},
            {"text": "Сколько раз в неделю у него занятия по языку?", "category": "morphology", "memory_indices": [0]},
            {"text": "Какой экзамен он планирует сдать?", "category": "synonym", "memory_indices": [1]},
            {"text": "Как он тренирует восприятие речи на слух?", "category": "low_overlap", "memory_indices": [2]},
        ],
    },
    {
        "id": "gym",
        "memories": [
            "Пользователь ходит в спортзал по вторникам и четвергам.",
            "Пользователь занимается с персональным тренером Игорем.",
            "Пользователь набирает мышечную массу и ест больше белка.",
        ],
        "queries": [
            {"text": "По каким дням пользователь ходит в спортзал?", "category": "exact", "memory_indices": [0]},
            {"text": "Как зовут тренера пользователя?", "category": "morphology", "memory_indices": [1]},
            {"text": "Какую цель он преследует в тренировках?", "category": "paraphrase", "memory_indices": [2]},
        ],
    },
    {
        "id": "home",
        "memories": [
            "Код от домофона в подъезде пользователя: 45К19.",
            "Пользователь снимает квартиру на третьем этаже без лифта.",
            "Пользователь планирует переезд в новую квартиру осенью.",
        ],
        "queries": [
            {"text": "Какой код домофона 45К19 в подъезде?", "category": "identifier", "memory_indices": [0]},
            {"text": "Есть ли в доме пользователя лифт?", "category": "synonym", "memory_indices": [1]},
            {"text": "Когда он собирается сменить жильё?", "category": "paraphrase", "memory_indices": [2]},
        ],
    },
    {
        "id": "delivery",
        "memories": [
            "Трек-номер посылки пользователя: TRK-5590231RU.",
            "Пользователь заказывает продукты с доставкой по субботам.",
            "Пользователь просит курьера оставлять заказ у двери.",
        ],
        "queries": [
            {"text": "Какой трек-номер TRK-5590231RU у посылки?", "category": "identifier", "memory_indices": [0]},
            {"text": "В какой день недели ему привозят продукты?", "category": "morphology", "memory_indices": [1]},
            {"text": "Должен ли курьер звонить в дверь или просто оставить заказ?", "category": "low_overlap", "memory_indices": [2]},
        ],
    },
    {
        "id": "garden",
        "memories": [
            "Пользователь выращивает помидоры на балконе.",
            "Пользователь поливает растения через день.",
            "У пользователя есть орхидея, которая цветёт раз в году.",
        ],
        "queries": [
            {"text": "Что выращивает пользователь на балконе?", "category": "exact", "memory_indices": [0]},
            {"text": "Как часто он поливает свои растения?", "category": "morphology", "memory_indices": [1]},
            {"text": "Часто ли цветёт орхидея пользователя?", "category": "paraphrase", "memory_indices": [2]},
        ],
    },
    {
        "id": "gaming",
        "memories": [
            "Пользователь проходит игру Baldur's Gate 3 по вечерам.",
            "У пользователя игровая консоль PlayStation 5.",
            "Пользователь состоит в игровом клане с друзьями по учёбе.",
        ],
        "queries": [
            {"text": "В какую игру играет пользователь по вечерам?", "category": "exact", "memory_indices": [0]},
            {"text": "Какая игровая приставка есть у пользователя?", "category": "synonym", "memory_indices": [1]},
            {"text": "С кем он играет в команде?", "category": "low_overlap", "memory_indices": [2]},
        ],
    },
    {
        "id": "eco",
        "memories": [
            "Пользователь сортирует мусор на пластик, стекло и бумагу.",
            "Пользователь ездит на работу на велосипеде, чтобы не пользоваться машиной.",
            "Пользователь отказался от одноразовых пакетов в магазинах.",
        ],
        "queries": [
            {"text": "Сортирует ли пользователь бытовые отходы?", "category": "exact", "memory_indices": [0]},
            {"text": "На чём он добирается до офиса вместо автомобиля?", "category": "morphology", "memory_indices": [1]},
            {"text": "Заботится ли он об экологии в быту?", "category": "low_overlap", "memory_indices": [2]},
        ],
    },
    {
        "id": "sleep",
        "memories": [
            "Пользователь ложится спать около полуночи в будние дни.",
            "Пользователь использует беруши, чтобы не просыпаться от шума.",
            "Пользователь просыпается без будильника по выходным.",
        ],
        "queries": [
            {"text": "Во сколько пользователь обычно ложится спать?", "category": "exact", "memory_indices": [0]},
            {"text": "Чем он затыкает уши, чтобы спать крепче?", "category": "synonym", "memory_indices": [1]},
            {"text": "Нужен ли ему будильник в субботу и воскресенье?", "category": "paraphrase", "memory_indices": [2]},
        ],
    },
    {
        "id": "cooking",
        "memories": [
            "Пользователь готовит борщ по бабушкиному рецепту.",
            "Пользователь не любит острую пищу.",
            "Пользователь печёт хлеб дома по выходным.",
        ],
        "queries": [
            {"text": "Чей рецепт борща использует пользователь?", "category": "exact", "memory_indices": [0]},
            {"text": "Нравится ли ему еда с перцем чили?", "category": "low_overlap", "memory_indices": [1]},
            {"text": "Что он выпекает по выходным?", "category": "morphology", "memory_indices": [2]},
        ],
    },
    {
        "id": "coffee",
        "memories": [
            "Пользователь заказывает капучино без сахара по утрам.",
            "Пользователь купил кофемолку с ручной настройкой помола.",
            "Пользователь избегает кофе после шести вечера, чтобы уснуть.",
            "Любимая кофейня пользователя находится рядом с домом.",
        ],
        "queries": [
            {"text": "Какой кофе пользователь берёт по утрам?", "category": "exact", "memory_indices": [0]},
            {"text": "Пьёт ли он кофе поздно вечером?", "category": "synonym", "memory_indices": [2]},
        ],
    },
    {
        "id": "photography",
        "memories": [
            "У пользователя беззеркальная камера Fujifilm X-T5.",
            "Пользователь снимает пейзажи на рассвете по выходным.",
            "Пользователь хранит фотографии в облачном архиве.",
            "Пользователь печатает лучшие кадры в фотоальбом раз в год.",
        ],
        "queries": [
            {"text": "Какая камера есть у пользователя?", "category": "exact", "memory_indices": [0]},
            {"text": "Где он предпочитает сохранять снимки после съёмки?", "category": "low_overlap", "memory_indices": [2]},
        ],
    },
    {
        "id": "cycling",
        "memories": [
            "У пользователя шоссейный велосипед Trek с карбоновой рамой.",
            "Пользователь катается по набережной по субботам утром.",
            "Пользователь готовится к любительской велогонке в сентябре.",
            "Пользователь всегда надевает шлем перед поездкой.",
        ],
        "queries": [
            {"text": "Какой у пользователя велосипед?", "category": "exact", "memory_indices": [0]},
            {"text": "К какому соревнованию он тренируется?", "category": "morphology", "memory_indices": [2]},
        ],
    },
    {
        "id": "meditation",
        "memories": [
            "Пользователь медитирует по 10 минут каждое утро.",
            "Пользователь использует приложение для дыхательных практик.",
            "Пользователь ведёт дневник благодарности перед сном.",
            "Пользователь проходил курс осознанности прошлым летом.",
        ],
        "queries": [
            {"text": "Сколько времени пользователь уделяет медитации утром?", "category": "morphology", "memory_indices": [0]},
            {"text": "Помогает ли ему что-то справляться со стрессом перед сном?", "category": "low_overlap", "memory_indices": [2]},
        ],
    },
    {
        "id": "podcasts",
        "memories": [
            "Пользователь слушает подкасты про технологии по дороге на работу.",
            "Пользователь подписан на еженедельный подкаст о космосе.",
            "Пользователь слушает подкасты на скорости 1.5x.",
        ],
        "queries": [
            {"text": "Что слушает пользователь по пути на работу?", "category": "exact", "memory_indices": [0]},
            {"text": "На какой скорости воспроизведения он слушает выпуски?", "category": "morphology", "memory_indices": [2]},
        ],
    },
    {
        "id": "volunteering",
        "memories": [
            "Пользователь раз в месяц волонтёрит в приюте для животных.",
            "Пользователь жертвует часть зарплаты на благотворительность.",
            "Пользователь участвовал в субботнике в своём дворе весной.",
        ],
        "queries": [
            {"text": "Где волонтёрит пользователь раз в месяц?", "category": "exact", "memory_indices": [0]},
            {"text": "Помогает ли он материально каким-то организациям?", "category": "low_overlap", "memory_indices": [1]},
        ],
    },
    {
        "id": "shopping",
        "memories": [
            "Пользователь предпочитает покупать одежду в секонд-хендах.",
            "Пользователь заказывает продукты оптом раз в две недели.",
            "Пользователь избегает импульсивных покупок в интернет-магазинах.",
        ],
        "queries": [
            {"text": "Где пользователь обычно покупает одежду?", "category": "exact", "memory_indices": [0]},
            {"text": "Легко ли ему удержаться от спонтанных покупок онлайн?", "category": "paraphrase", "memory_indices": [2]},
        ],
    },
    {
        "id": "locker",
        "memories": [
            "Код от шкафчика пользователя в спортзале: 3319.",
            "Пользователь хранит запасной комплект одежды в шкафчике.",
            "Пользователь меняет замок на шкафчике раз в год.",
        ],
        "queries": [
            {"text": "Какой код 3319 от шкафчика в зале?", "category": "identifier", "memory_indices": [0]},
            {"text": "Что лежит у него запасное в шкафчике?", "category": "morphology", "memory_indices": [1]},
        ],
    },
]

# Queries with no ground truth in this dataset at all — deliberately about
# things nothing above ever mentions. Used to measure how confidently (or
# not) each retriever scores something that just isn't there; a good
# retriever's negatives should score low, not merely "lower than positives".
NEGATIVE_QUERIES = [
    "Какая столица у Австралии?",
    "Сколько будет 15 умножить на 12?",
    "Пользователь занимается подводным плаванием с аквалангом?",
    "Есть ли у пользователя лицензия пилота?",
    "Какой у него любимый вид борьбы сумо?",
    "Собирает ли пользователь марки?",
    "Играет ли пользователь на волынке?",
    "Есть ли у него ферма лам?",
    "Изучает ли пользователь древнегреческий язык?",
    "Разводит ли пользователь пчёл на даче?",
]
