# plogs — Pretty Logs


### Project Goals
In the beginning, the goal of Pretty Logs was to create a colorful logging system designed for scalable projects. Later,
my vision for Pretty Logs widened in scope to also include useful debugging tools that are found in JavaScript and
aren't that readily available in Python.

In the future, we anticipate better control to configure logging settings and a debugging log level where debug tools
can only print to.


### Installation
The easiest way to install Pretty Logs is to install via pip.

```
$ pip3 install -U plogs
```

Is also possible to download and import Pretty Logs directly into your project. If you are you looking to do so, I
recommend cloning the GitHub repository to ensure the plogs module hierarchy is kept the same.

```
$ git clone https://github.com/11/plogs.git
```


### Setup

Importing Pretty Logs into your project is quite simple. All that's required is to <b>import plogs</b> and instantiate an instance of <b>Logger</b>.

```python3
from plogs import get_logger
logging = get_logger()
```

It's recommend to do the following steps inside a `__init__.py` file at the root level of your project so Pretty Logs can be referenced throughout the entire application.


### Log With Colors

Pretty Logs' main feature is color coding different logging levels and statuses. The default logging levels are set mapped to:

| Log Level         | Color |
| ---               | --- |
| logging.info	    | gray |
| logging.status	| bold |
| logging.success	| green |
| logging.warning	| orange |
| logging.error     | red |
| logging.critical	| red highlights |


### Log to Files
Pretty Logs can write colored logs to files. This is done through Pretty Logs' <b>config</b> function.

```python3
from plogs import get_logger

logging = get_logger()
logging.config(to_file=True)

logging.info('this is will be written to a file')
```

By default, files are written to `/var/log/plogs/plog_01.log`. `/var/log/` is chosen as the default directory because it is commonly used on unix based machines and in
docker based services.

If you are looking to use another filename and location, it can simply be edited like such:

```python3
logging.config(to_file=True, file_location='your/filepath/here/', filename='new_file.log')
```


### Format Your Logs

Pretty Logs allows for a lot of customization. This can be done by editing the logging <b>config</b> and supplying Pretty Logs with a formatted string.

The following are all the configurable variables:


| Variable        | Type   | Description |
| ---             | ---    | ---         |
| `pretty`        | `bool` | Setting to `True` will add color to logs, `False` will un-color logs |
| `show_levels`   | `bool` | Setting to `True` will show logging level in formatted log, `False` show no logging level |
| `show_time`     | `bool` | Setting to `True` will show time in formatted log, `False` doesn't show time |
| `to_file`       | `bool` | Setting to `True` writes logs to `file_location`, `False` writes to `standard output` |
| `file_location` | `str`  | Default `file_location` is `/var/log/plogs/`, otherwise a file location of your choice |
| `filename`      | `str`  | Default log file is `plog_01.log`, otherwise a filename of your choice |


An example of a formatted logs would be like such:

```python3
from plogs import get_logger

logging = get_logger()

# configure plogs to allow logging level and date/time
logging.config(show_levels=True, show_time=True)

# once configged logs with `{level}` to show the logging level, and `{time}` to show the datetime the log was written
logging.format('[{level}] - {time} - {msg}')

# finally write logs
logging.status('Show me the logs')
logging.info('We got some info')

# Output:
# [STATUS] - 2018-12-11 11:56:05 - Show me the logs
# [INFO] - 2018-12-11 11:56:09 - We got some info
```


### Log Tables
```python3
from plogs import get_logger
logging = get_logger()

class Example:

    def __init__(self, a, b):
        self.a = a
        self.b = b


ex1 = Example(1, 2)
ex2 = Example('a', 'b')

logging.table(ex1, ex2)
```
The output would be like
```
+ --------------- +
|     |  a  |  b  |
+ --- | --- | --- +
| ex1 |  1  |  2  |
+ --- | --- | --- |
| ex2 | 'a' | 'b' |
+ --------------- +
```
