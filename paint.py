import matplotlib.pyplot as plt

# x = [
#     'ORG', 'ORD', 'MON', 'LOC', 'PER', 'INT', 'FRE', 'DAT', 'TIM', 'DUR',
#     'LEN', 'FRA', 'AGE', 'WEI', 'MEA'
# ]
x = [
    'ORG', 'LOC', 'ORD', 'TIM', 'PER', 'INT', 'PER', 'DAT', 'DUR', 'FRA',
    'FRE', 'DEC', 'MEA', 'ANG'
]
y1 = [
    81.99, 82.76, 84.57, 75.91, 97.14, 90.97, 82.11, 88.2, 88.42, 92.86, 91.23,
    80, 87.5, 85.71
]
y2 = [
    82.61, 84.03, 83.5, 74.81, 100, 92.23, 86.69, 90.53, 89.97, 96, 96.55, 100,
    93.33, 100
]
plt.bar(x, y2, label='MSRA_our', fc=(0.97, 0.79, 0.08))
plt.bar(x, y1, label='MSRA_baseline', fc=(0.36, 0.55, 0.78))

plt.xticks(x, x, rotation=90)
# plt.bar(x, y1, width=0.7, label='MSRA_baseline')
# plt.bar(x, y2, width=0.7, label='MSRA_ours')
plt.legend()
plt.show()
