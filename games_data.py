# Used for gathering games data from storage
import os
import json
import csv

import steam_api

import numpy as np


class GamesData:
    def __init__(self, dir_users, dir_games, file_ledger, file_users_conv, use_recent,
                 min_games_played):
        self.dir_users = dir_users
        self.dir_games = dir_games
        self.file_ledger = file_ledger
        self.file_users_conv = file_users_conv
        self.min_games_played = min_games_played

        self.use_recent = use_recent

        print('\tLoading Original Games Mapping')
        self.games_map = self.__load_games()

        print('\tLoading Ledger')
        self.ledger_to, self.ledger_from = self.__get_ledger()

        self.games_count = len(self.ledger_to)
        print('\tConverting Users')
        self.users_conv = self.__get_users_conv()

    # Creates a 1 to 1 mapping for all game objects to a 0 based index
    def __make_ledger(self):
        i = 0
        ledger_to = {}
        for k in self.games_map:
            ledger_to[k] = i
            i += 1

        json.dump(ledger_to, open(self.file_ledger, 'w+'))
        return ledger_to

    def __load_games(self):
        games = {}
        for f in os.listdir(self.dir_games):
            game = steam_api.Game(json.load(open(os.path.join(self.dir_games, f), "r+")))
            games[game.appid] = game
        return games

    def __load_ledger(self):
        ledger_to = json.load(open(self.file_ledger, "r+"))
        return ledger_to

    def __invert_ledger(self, ledger_to):
        ledger_from = {}
        for (k, v) in ledger_to.items():
            ledger_from[v] = k
        return ledger_from

    def __get_ledger(self):
        ledger_to = self.__load_ledger() if os.path.exists(self.file_ledger) else self.__make_ledger()
        return ledger_to, self.__invert_ledger(ledger_to)

    def __count_games_played(self, games):
        return np.argwhere(games > 0).flatten().shape[0]

    def get_user_games_from_file(self, steamid):
        dat = json.load(open(os.path.join(self.dir_users, steamid), 'r+'))
        usr_games = np.array([0] * self.games_count)
        for (appid, g) in dat['games'].items():
            usr_games[self.ledger_to[appid]] = (g['playtime_2weeks'] if 'playtime_2weeks' in g else 0) \
                if self.use_recent else g['playtime_forever']
        del dat
        return usr_games

    def __make_users_conv(self):
        fi = open(self.file_users_conv, 'w+', newline='')
        stream_out = csv.writer(fi)

        for f in os.listdir(self.dir_users):
            games = self.get_user_games_from_file(f)
            if self.__count_games_played(games) > self.min_games_played:
                stream_out.writerow(games)

        fi.close()

    def __get_users_conv(self):
        if not os.path.exists(self.file_users_conv):
            self.__make_users_conv()
        f = open(self.file_users_conv, 'r+')
        return f, csv.reader(f, delimiter=',', lineterminator='\r\n')

    def get_game_from_index(self, index):
        return self.games_map[self.ledger_from[index]]

    def get_users_conv(self):
        return self.users_conv
