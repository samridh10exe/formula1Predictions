# Formula One Race Time Predictions: 2025 Saudi Arabian Grand Prix

This file contains the race time predictions for the 2025 Saudi Arabian Grand Prix at Jeddah Corniche Circuit from various models. All lap times have been converted to estimated full race times (50 laps) in hours:minutes:seconds format.

## Model 1: Feature-Rich GBR Model (Original)

```
                   Driver            Predicted Race Time
0         Charles Leclerc             1:16:13.48
1          Lewis Hamilton             1:16:13.48
2           Oscar Piastri             1:16:13.89
3          Max Verstappen             1:16:13.89
4          George Russell             1:16:13.89
5   Andrea Kimi Antonelli             1:16:13.89
6            Yuki Tsunoda             1:16:13.89
7            Pierre Gasly             1:16:13.89
8          Oliver Bearman             1:16:13.89
9            Lando Norris             1:16:13.89
10            Liam Lawson             1:16:13.89
11           Isack Hadjar             1:16:13.89
12      Gabriel Bortoleto             1:16:13.89
13           Esteban Ocon             1:16:13.89
14        Nico Hülkenberg             1:16:13.89
15            Jack Doohan             1:16:13.89
16        Fernando Alonso             1:16:14.07
17           Lance Stroll             1:16:14.07
18        Alexander Albon             1:16:14.16
19       Carlos Sainz Jr.             1:16:14.16
```

## Model 2: Qualifying-Only Model (Simple)

```
                   Driver            Predicted Race Time
0         Charles Leclerc             1:15:06.27
1          Kimi Antonelli             1:15:06.27
2          Max Verstappen             1:15:08.35
3           Oscar Piastri             1:15:19.29
4            Yuki Tsunoda             1:16:11.26
5         Alexander Albon             1:16:25.37
6          Nico Hulkenberg             1:16:32.99
7            Jack Doohan             1:16:47.30
8            Lando Norris             1:16:59.57
9           Carlos Sainz             1:17:06.37
10      Gabriel Bortoleto             1:17:12.82
11           Esteban Ocon             1:17:12.82
12         George Russell             1:17:15.81
13         Lewis Hamilton             1:17:27.90
14            Liam Lawson             1:18:03.23
15           Pierre Gasly             1:18:19.75
16           Isack Hadjar             1:18:19.75
17           Lance Stroll             1:18:25.41
18        Fernando Alonso             1:18:29.90
19         Oliver Bearman             1:21:28.70
```

## Ridge Regression Model (Best MAE: 0.0048)

```
                   Driver            Predicted Race Time
0          George Russell             0:00:58.10
1   Andrea Kimi Antonelli             0:00:58.10
2          Lewis Hamilton             0:00:58.25
3         Charles Leclerc             0:00:58.25
4         Fernando Alonso             0:00:58.36
5            Lance Stroll             0:00:58.36
6            Yuki Tsunoda             0:00:58.42
7          Max Verstappen             0:00:58.42
8          Oliver Bearman             0:00:58.42
9            Isack Hadjar             0:00:58.42
10            Liam Lawson             0:00:58.42
11        Nico Hülkenberg             0:00:58.42
12           Esteban Ocon             0:00:58.42
13      Gabriel Bortoleto             0:00:58.42
14           Lando Norris             0:00:58.46
15          Oscar Piastri             0:00:58.46
16       Carlos Sainz Jr.             0:00:58.56
17        Alexander Albon             0:00:58.56
18           Pierre Gasly             0:00:58.62
19            Jack Doohan             0:00:58.62
```
*Note: The Ridge Regression values appear unrealistically low and likely need recalibration.*

## Gradient Boosting Model (MAE: 0.0548)

```
                   Driver            Predicted Race Time
0         Charles Leclerc             1:16:13.48
1          Lewis Hamilton             1:16:13.48
2           Oscar Piastri             1:16:13.89
3          Max Verstappen             1:16:13.89
4          George Russell             1:16:13.89
5   Andrea Kimi Antonelli             1:16:13.89
6            Yuki Tsunoda             1:16:13.89
7            Pierre Gasly             1:16:13.89
8          Oliver Bearman             1:16:13.89
9            Lando Norris             1:16:13.89
10            Liam Lawson             1:16:13.89
11           Isack Hadjar             1:16:13.89
12      Gabriel Bortoleto             1:16:13.89
13           Esteban Ocon             1:16:13.89
14        Nico Hülkenberg             1:16:13.89
15            Jack Doohan             1:16:13.89
16        Fernando Alonso             1:16:14.07
17           Lance Stroll             1:16:14.07
18        Alexander Albon             1:16:14.16
19       Carlos Sainz Jr.             1:16:14.16
```

## XGBoost Model (MAE: 0.0698)

```
                   Driver            Predicted Race Time
0          Max Verstappen             1:16:23.83
1           Oscar Piastri             1:16:23.83
2          George Russell             1:16:23.83
3         Charles Leclerc             1:16:23.83
4   Andrea Kimi Antonelli             1:16:23.83
5        Carlos Sainz Jr.             1:16:23.83
6          Lewis Hamilton             1:16:23.83
7            Yuki Tsunoda             1:16:23.83
8            Lando Norris             1:16:23.83
9         Alexander Albon             1:16:23.83
10            Liam Lawson             1:16:23.83
11           Isack Hadjar             1:16:23.83
12         Oliver Bearman             1:16:23.83
13      Gabriel Bortoleto             1:16:23.83
14           Esteban Ocon             1:16:23.83
15        Nico Hülkenberg             1:16:23.83
16        Fernando Alonso             1:16:23.85
17           Lance Stroll             1:16:23.85
18           Pierre Gasly             1:16:23.88
19            Jack Doohan             1:16:23.88
```

## LightGBM Model (MAE: 0.0903)

```
                   Driver            Predicted Race Time
0           Oscar Piastri             1:16:30.67
1            Lando Norris             1:16:30.67
2          Lewis Hamilton             1:16:31.72
3         Charles Leclerc             1:16:31.72
4            Yuki Tsunoda             1:16:31.77
5            Isack Hadjar             1:16:31.77
6             Liam Lawson             1:16:31.77
7          Max Verstappen             1:16:31.77
8          Oliver Bearman             1:16:31.77
9            Lance Stroll             1:16:31.77
10        Fernando Alonso             1:16:31.77
11        Nico Hülkenberg             1:16:31.77
12           Esteban Ocon             1:16:31.77
13      Gabriel Bortoleto             1:16:31.77
14  Andrea Kimi Antonelli             1:16:31.81
15         George Russell             1:16:31.81
16       Carlos Sainz Jr.             1:16:31.85
17        Alexander Albon             1:16:31.85
18           Pierre Gasly             1:16:32.65
19            Jack Doohan             1:16:32.65
```
