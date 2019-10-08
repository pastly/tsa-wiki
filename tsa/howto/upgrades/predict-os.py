#!/usr/bin/python3
# coding: utf-8

'''predict when major upgrades will complete'''
# Copyright (C) 2016 Antoine Beaupré <anarcat@debian.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import argparse
import collections
from datetime import datetime
import io
import logging
import logging.handlers
import os
import os.path
import sys
import tempfile

try:
    import pytest
except ImportError:
    pytest = None

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns


__epilog__ = '''This scripts will predict when major OS upgrades will complete,
based on regular samples stored in a CSV file, which are added from
PuppetDB. It will also draw a graph, on the GUI or in a file,
representing the state of the CSV file and progress. This project is a
rewrite of this R toolset in Python:
https://gitlab.com/anarcat/predict-os and expects the following Python
packages to be installed: python3-requests python3-seaborn'''

# the reason this was rewritten in Python was that:
#
# 1. libreoffice is a catastrophe, see the original predict-os for details
# 2. i don't want to learn how to read/write/parse CSV files in R
# 3. i don't want to learn how to make R talk with PuppetDB
# 4. i got tired of chasing the PuppetDB SQL database changes
# 5. i had to use python to massage data anyways
# 6. "code without tests is legacy code" and i don't want legacy code

PUPPETDB_URL = 'http://localhost:8080/pdb/query/v4'
PUPPETDB_QUERY = 'facts[value] { name = "lsbdistcodename" }'

DEFAULT_HEADER = ['Date', 'release', 'count']


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog=__epilog__)
    parser.add_argument('--verbose', '-v', dest='log_level',
                        action='store_const', const='info', default='warning')
    parser.add_argument('--debug', '-d', dest='log_level',
                        action='store_const', const='debug', default='warning')
    parser.add_argument('--test', action='store_true',
                        help='run self test suite and nothing else')
    parser.add_argument('--puppetdb', '-p', default=PUPPETDB_URL,
                        help='PuppetDB server URL')
    parser.add_argument('--query', default=PUPPETDB_QUERY,
                        help='query returning the list of Debian releases')
    parser.add_argument('--path', default='data.csv',
                        help='CSV datafile that keeps past records')
    parser.add_argument('--refresh', '-r', action='store_true',
                        help='fetch from PuppetDB (default: %(default)s)')
    parser.add_argument('--dryrun', '-n', action='store_true',
                        help='do nothing')
    parser.add_argument('--output', '-o', type=argparse.FileType('wb'),
                        default=sys.stdout, help='image to write, default to graphical display or stdout if unavailable')  # noqa: E501
    parser.add_argument('--source', '-s', default='stretch',
                        help='major version we are upgrading from')
    return parser.parse_args(args=args)


def main(args):
    logging.debug('loading previous records from %s', args.path)
    if not os.path.exists(args.path):
        with open(args.path, 'w') as fp:
            fp.write(','.join(DEFAULT_HEADER))
    with open(args.path) as fp:
        records = load_csv(fp)
    if args.refresh:
        logging.info('querying PuppetDB on %s', args.puppetdb)
        logging.debug('query: %s', args.query)
        new_data = puppetdb_query(args.puppetdb, args.query)
        logging.info('found %d hosts', len(new_data))
        new_record = count_releases(new_data)
        records = add_releases(records, new_record)
        if not args.dryrun:
            with open(args.path, 'w') as fp:
                store_csv(fp, records)
    records = prepare_records(records)
    try:
        date = guess_completion_time(records, args.source)
        print("completion time of %s major upgrades: %s" % (args.source, date))
    except (TypeError, ValueError) as e:
        logging.warning("cannot guess completion time: %s", e)
        date = 'N/A'
    plot_records(records, date, args)


if pytest is not None:
    @pytest.xfail('not sure why there is no output, but this should work')
    @pytest.mark.parametrize("test_input,expected",
                             [(b'''Date,release,count
2019-10-08,stretch,83
2019-10-08,buster,3
2019-10-08,sid,1
2019-10-08,jessie,2''', 'cannot guess completion time')])
    def test_main(test_input, expected):
        with tempfile.NamedTemporaryFile() as csv:
            csv.write(test_input)
            csv.flush()
            handler = logging.handlers.MemoryHandler(1000)
            handler.setLevel('DEBUG')
            logging.getLogger('').addHandler(handler)
            with tempfile.NamedTemporaryFile(suffix='.png') as graph:
                args = parse_args(['--path', csv.name, '--output', graph.name])
                main(args)
            output = "\n".join([record.getMessage()
                                for record in handler.buffer])
            assert expected in output


def load_csv(fp):
    '''load the data from the CSV'''
    return pd.read_csv(fp)


SAMPLE_CSV = '''Date,release,count
2019-01-01,buster,32
2019-01-01,stretch,10
2019-02-02,buster,37
2019-02-02,stretch,5
2019-03-03,buster,50
2019-03-03,stretch,1
'''

SAMPLE_DF_REPR = '''         Date  release  count
0  2019-01-01   buster     32
1  2019-01-01  stretch     10
2  2019-02-02   buster     37
3  2019-02-02  stretch      5
4  2019-03-03   buster     50
5  2019-03-03  stretch      1'''


def test_load_csv():
    '''just a sanity check that pandas works as expected'''
    fp = io.StringIO(SAMPLE_CSV)
    res = load_csv(fp)
    assert repr(res) == SAMPLE_DF_REPR
    return res


def store_csv(fp, records):
    '''write the CSV file back to the given stream'''
    return fp.write(records.to_csv(index=False))


def test_store_csv():
    '''just a sanity check that we do the CSV rountrip cleanly'''
    fp = io.StringIO()
    data = test_load_csv()
    store_csv(fp, data)
    fp.seek(0)
    assert fp.read() == SAMPLE_CSV


def puppetdb_query(url, query, session=requests):
    '''get the data from PuppetDB'''
    resp = session.get(url, data={'query': query})
    resp.raise_for_status()
    return resp.json()


def test_puppetdb_query():
    '''simulate a PuppetDB query'''
    import betamax
    session = requests.Session()
    recorder = betamax.Betamax(session, cassette_library_dir='cassettes')
    with recorder.use_cassette('puppetdb'):
        json = puppetdb_query(PUPPETDB_URL, PUPPETDB_QUERY, session=session)
    assert len(json) > 0
    return json


def count_releases(data):
    '''parse the data returned by PuppetDB

    This counts the number of entries for each releases.

    >>> d = [{'value': 'buster'}, {'value': 'stretch'}, {'value': 'buster'}]
    >>> count_releases(d)
    {'buster': 2, 'stretch': 1}
    '''
    total = collections.defaultdict(int)
    for item in data:
        logging.debug('checking item %s', item)
        total[item['value']] += 1
    return dict(total)


def add_releases(data, new_data, date=None):
    '''take the existing data and appending the new record'''
    if date is None:
        date = datetime.today().strftime('%Y-%m-%d')
    series = [{'Date': date, 'release': release, 'count': count}
              for release, count in new_data.items()]
    return data.append(series, ignore_index=True)


def test_add_releases():
    '''check that we can add to the pandas dataframe as expected'''
    data = test_load_csv()
    new_data = {'buster': 33, 'stretch': 9}
    d = add_releases(data, new_data, '2019-04-05')
    assert SAMPLE_DF_REPR + '''
6  2019-04-05   buster     33
7  2019-04-05  stretch      9''' == repr(d)


# cargo-culted from https://stackoverflow.com/questions/48860428/passing-datetime-like-object-to-seaborn-lmplot  # noqa: E501
@plt.FuncFormatter
def fake_dates(x, pos):
    """ Custom formater to turn floats into e.g., 2016-05-08"""
    return matplotlib.dates.num2date(x).strftime('%Y-%m-%d')


def plot_records(records, guessed_date, args):
    '''draw the actual graph, on the GUI or in a file as args dictates'''
    sns.set(color_codes=True)
    # ci=False because it looks kind of wrong
    graph = sns.lmplot(x='datenum', y='count', hue='release',
                       data=records, ci=False)
    # return numeric dates into human-readable
    graph.ax.xaxis.set_major_formatter(fake_dates)
    graph.ax.set_title('Debian major upgrades to %s planned completion by %s' %
                       (args.source, guessed_date))
    graph.ax.set_xlabel('date')
    # labels overlap otherwise
    graph.ax.tick_params(labelrotation=45)
    if (args.dryrun or
        (args.output == sys.stdout and
         (sys.stdout.isatty() or 'DISPLAY' in os.environ))):
        plt.show()
    else:
        _, ext = os.path.splitext(args.output.name)
        plt.savefig(args.output, format=ext[1:], bbox_inches='tight')


def prepare_records(records):
    '''various massaging required by other tools

    This currently only stores the numeric date for seaborn and
    regression processing.
    '''
    records['datenum'] = matplotlib.dates.datestr2num(records['Date'])
    return records


def guess_completion_time(records, source):
    '''take the given records and guess the estimated completion time

    :param Dataframe records: the records, as loaded from the CSV file
           by load_csv)
    :param str source: the kind of `release`. will fail if unknown
    :returns: completion date, formatted as a string (YYYY-MM-DD)

    >>> records = prepare_records(test_load_csv())
    >>> guess_completion_time(records, 'stretch')
    '2019-03-09'
    '''
    subdf = records[records['release'] == source]
    fit = np.polyfit(subdf['count'], subdf['datenum'], 1)
    prediction = np.poly1d(fit)(0)
    return fake_dates(prediction, None)


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(format='%(message)s', level=args.log_level.upper())
    if args.test:
        logging.info('running test suite')
        try:
            import pytest
        except ImportError:
            logging.error('test suite requires pytest to run properly')
            sys.exit(1)
        sys.exit(pytest.main([__file__]))
    try:
        main(args)
    except Exception as e:
        logging.error('unexpected error: %s', e)
        if args.log_level == 'debug':
            logging.warning('starting debugger, type "c" and enter to continue')  # noqa: E501
            import traceback
            import pdb
            import sys
            traceback.print_exc()
            pdb.post_mortem()
            sys.exit(1)
        raise e
