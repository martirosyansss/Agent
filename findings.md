# Findings & Decisions

## Requirements
- Пользователь запросил точный поэтапный план реализации проекта по фазам из ТЗ.
- План должен быть составлен с использованием planning skill, а не просто ответом в чате.
- План должен учитывать текущую версию ТЗ V1.5, а не только ранний roadmap 0-8.
- План должен оставаться практическим: от MVP до live, без преждевременного усложнения.

## Research Findings
- Planning skill требует создать `task_plan.md`, `findings.md`, `progress.md` в корне проекта и вести в них состояние задачи.
- В ТЗ есть базовый roadmap: Этапы 0-8, от фундамента до live trading.
- В V1.1 и V1.5 добавлены обязательные production-модули: Circuit Breakers, Watchdog, Data Integrity Guard, Anti-Corruption Layer, Trade Analyzer.
- Переход `paper -> live` допускается только при выполнении жёстких условий: Win Rate > 50% за 7 дней, PnL > 0, Max Drawdown < 5%, минимум 50 завершённых сделок, отсутствие критических ошибок, ручное подтверждение пользователя.
- ТЗ требует запускать систему в двух процессах: основной `main.py` и отдельный `watchdog.py`.
- ТЗ задаёт целевую структуру проекта `sentinel/` с отдельными пакетами `core`, `collector`, `database`, `features`, `strategy`, `risk`, `execution`, `position`, `guards`, `analyzer`, `backtest`, `telegram_bot`, `dashboard`, `tests`.
- ТЗ прямо запрещает трактовать лимиты как абсолютную математическую гарантию: live должен опираться на exchange-native protective orders.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| План расширен сверх Этапов 0-8 до полной production-sequence | В версии 1.5 есть обязательные модули, которых нет в коротком roadmap |
| Сначала одна базовая стратегия, потом strategy arsenal | Это снижает объём одновременных рисков и ускоряет получение baseline-метрик |
| Risk Hardening поставлен до live rollout | Без этого не выполняются критерии безопасности из ТЗ |
| Trade Analyzer разделён на Level 1, Level 2, Level 3 | Так его проще включать по зрелости данных и инфраструктуры |
| ML вынесен в позднюю фазу и начинается с shadow mode | Это соответствует критериям V1.5 и снижает риск ложной автоматизации |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Базовый roadmap ТЗ не покрывает все обязательные модули V1.1/V1.5 | План расширен до полного пути внедрения, сохраняя исходные этапы как основу |
| В проекте не было planning-файлов | Файлы созданы в корне проекта согласно skill |

## Resources
- `TECHNICAL_SPECIFICATION.md`
- `c:\Users\marti\.agents\skills\planning-with-files\SKILL.md`
- `c:\Users\marti\.agents\skills\planning-with-files\templates\task_plan.md`
- `c:\Users\marti\.agents\skills\planning-with-files\templates\findings.md`
- `c:\Users\marti\.agents\skills\planning-with-files\templates\progress.md`

## Visual/Browser Findings
- Не использовались.