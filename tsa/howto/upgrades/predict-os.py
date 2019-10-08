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
import logging
import sys

import betamax
import pandas as pd
import requests
import seaborn as sns


__epilog__ = '''This scripts will predict when major OS upgrades will complete,
based on regular samples stored in a CSV file, which are added from
PuppetDB.'''

PUPPETDB_URL = 'http://localhost:8080/pdb/query/v4'
PUPPETDB_QUERY = 'facts[value] { name = "lsbdistcodename" }'


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog=__epilog__)
    parser.add_argument('--verbose', '-v', dest='log_level',
                        action='store_const', const='info', default='warning')
    parser.add_argument('--debug', '-d', dest='log_level',
                        action='store_const', const='debug', default='warning')
    parser.add_argument('--selftest', '-s', action='store_true',
                        help='run self test suite and nothing else')
    parser.add_argument('--puppetdb', '-p', default=PUPPETDB_URL,
                        help='PuppetDB server URL')
    parser.add_argument('--query', default=PUPPETDB_QUERY,
                        help='query returning the list of Debian releases')
    parser.add_argument('--path', default='data.csv',
                        help='CSV datafile that keeps past records')
    parser.add_argument('--refresh', '-r', action='store_true',
                        help='pull current results from PuppetDB (default: %(default)s)')
    parser.add_argument('--dryrun', '-n', action='store_true',
                        help='do nothing')
    return parser.parse_args(args=args)


def main(args):
    logging.debug('loading previous records from %s', args.path)
    with open(args.path) as fp:
        records = load_csv(fp)
    if args.refresh:
        logging.info('querying PuppetDB on %s', args.puppetdb)
        logging.debug('query: %s', args.query)
        new_data = puppetdb_query(args.puppetdb, args.query)
        logging.info('found %d hosts', len(new_data))
        new_record = count_releases(new_data)
        records = add_releases(records, new_record)
    plot_records(records)


def load_csv(fp):
    '''load the data from the CSV, indexed by date'''
    return pd.read_csv(fp)


def test_load_csv():
    import io
    data = b'''Date,release,count
2019-01-01,buster,32
2019-01-01,stretch,10
2019-02-02,buster,37
2019-02-02,stretch,5
2019-03-03,buster,50
2019-03-03,stretch,1
'''
    fp = io.StringIO(data.decode('utf-8'))
    res = load_csv(fp)
    assert repr(res) == '''         Date  release  count
0  2019-01-01   buster     32
1  2019-01-01  stretch     10
2  2019-02-02   buster     37
3  2019-02-02  stretch      5
4  2019-03-03   buster     50
5  2019-03-03  stretch      1'''  # noqa: W291
    return res


def puppetdb_query(url, query, session=requests):
    '''get the data from PuppetDB'''
    resp = session.get(url, data={'query': query})
    resp.raise_for_status()
    return resp.json()


def test_puppetdb_query():
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
    data = test_load_csv()
    new_data = {'buster': 33, 'stretch': 9}
    d = add_releases(data, new_data, '2019-04-05')
    assert '''         Date  release  count
0  2019-01-01   buster     32
1  2019-01-01  stretch     10
2  2019-02-02   buster     37
3  2019-02-02  stretch      5
4  2019-03-03   buster     50
5  2019-03-03  stretch      1
6  2019-04-05   buster     33
7  2019-04-05  stretch      9''' == repr(d)


def plot_records(records):
    sns.lmplot(x='Date', y='count', hue='release', data=records)


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(format='%(message)s', level=args.log_level.upper())
    if args.selftest:
        logging.info('# running test suite')
        test_puppetdb_query()
        sys.exit(0)
    try:
        main(args)
    except Exception as e:
        logging.error('unexpected error: %s', e)
        if args.log_level == 'debug':
            logging.warning('starting debugger, type "c" and enter to continue...')
            import traceback
            import pdb
            import sys
            traceback.print_exc()
            pdb.post_mortem()
            sys.exit(1)
        raise e
