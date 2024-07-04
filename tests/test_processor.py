"""Test the module :py:mod:`~src.processor`."""

import gc

import pandas as pd
import pytest

from src.processor import RoboticVacuumCleaners, SmartThermostats, TraditionalVacuumCleaners

preprocd_prt_msg = [
    'Preprocessing the preparatory data ... ',
    '\tRemoving short reviews (No. of words < 20) ... Done.',
    '\tRemoving non-English reviews ... Done.',
    '\tDetermining sentiment on rating ... Done.',
    '\tCalculating VADER sentiment scores ... Done.',
    '\tDetermining sentiment on VADER sentiment score ... Done.',
    '\tProcessing review text ... ',
    '\t\tRemoving stopwords ... Done.',
    '\t\tRemoving punctuation ... Done.',
    '\t\tRemoving single letters ... Done.',
    '\t\tRemoving digits ... Done.',
    '\t\tLemmatizing texts ... Done.',
]


class TestRoboVac:

    @pytest.mark.parametrize('use_db', [True, False])
    def test_load_raw_data(self, capfd, use_db):
        rvc = RoboticVacuumCleaners(load_preprocd_data=False, use_db=use_db)
        rvc.load_raw_data(update=True, verbose=True)
        out, _ = capfd.readouterr()
        assert f"Reading the raw data of {rvc.PRODUCT_NAME.lower()} reviews ... " in out

        if use_db:
            assert 'Importing data into the table "amazon_reviews"."vacuum_cleaners_robotic"' in out

            rvc = RoboticVacuumCleaners(load_preprocd_data=False, use_db=use_db)
            rvc.load_raw_data(verbose=True)
            out, _ = capfd.readouterr()
            assert (f"Loading the raw data of the {rvc.PRODUCT_NAME.lower()} reviews ... " in out
                    and "Done." in out)

            rvc.load_raw_data(verbose=True)
            out, _ = capfd.readouterr()
            assert "The raw data is already loaded." in out

        else:
            assert '"raw_data.pkl"' in out and "Done." in out

            rvc = RoboticVacuumCleaners(load_preprocd_data=False, use_db=use_db)
            rvc.load_raw_data(verbose=True)
            out, _ = capfd.readouterr()
            assert "Loading " in out and "raw_data.pkl" in out and "Done." in out

        assert isinstance(rvc.raw_data, pd.DataFrame)
        assert len(rvc.raw_data) >= 143217

        del rvc
        gc.collect()

    @pytest.mark.parametrize('use_db', [True, False])
    def test_load_prep_data(self, capfd, use_db):
        rvc = RoboticVacuumCleaners(load_preprocd_data=False, use_db=use_db)
        rvc.load_prep_data(update=True, verbose=True)
        out, _ = capfd.readouterr()
        assert "Making data ready for preprocessing ... " in out

        if use_db:
            assert ('Importing data into the table '
                    '"amazon_reviews"."vacuum_cleaners_robotic_prep"') in out

            rvc = RoboticVacuumCleaners(load_preprocd_data=False, use_db=use_db)
            rvc.load_prep_data(verbose=True)
            out, _ = capfd.readouterr()
            assert (f"Loading the preparatory data of the {rvc.PRODUCT_NAME.lower()} reviews ... "
                    in out and "Done." in out)

            rvc.load_prep_data(verbose=True)
            out, _ = capfd.readouterr()
            assert 'The preparatory data is already loaded.' in out

        else:
            assert '"prep_data.pkl"' in out and "Done." in out

            rvc = RoboticVacuumCleaners(load_preprocd_data=False)
            rvc.load_prep_data(verbose=True)
            out, _ = capfd.readouterr()
            assert "Loading " in out and "prep_data.pkl" in out and "Done." in out

        assert isinstance(rvc.prep_data, pd.DataFrame)
        assert len(rvc.prep_data) >= 143217

        del rvc
        gc.collect()

    @pytest.mark.parametrize('use_db', [True, False])
    @pytest.mark.parametrize('verified_reviews_only', [True, False])
    @pytest.mark.parametrize('dual_scale', [True, False])
    def test_load_preprocd_data(self, capfd, use_db, verified_reviews_only, dual_scale):
        rvc = RoboticVacuumCleaners(load_preprocd_data=False, use_db=use_db)
        rvc.load_preprocd_data(update=True, verbose=True)
        out, _ = capfd.readouterr()
        assert '\n'.join(preprocd_prt_msg) in out.replace('\x1b[0m', '')

        if use_db:
            assert ('Importing data into the table '
                    '"amazon_reviews"."vacuum_cleaners_robotic_preprocd"') in out

            rvc = RoboticVacuumCleaners(load_preprocd_data=False, use_db=use_db)
            rvc.load_preprocd_data(verbose=True)
            out, _ = capfd.readouterr()
            assert "Loading the preprocessed data of product reviews ... " in out and "Done." in out

            rvc.load_preprocd_data(verbose=True)
            out, _ = capfd.readouterr()
            assert 'The preprocessed data is already loaded.' in out

        else:
            assert '"preprocd_data.pkl"' in out and "Done." in out

            rvc = RoboticVacuumCleaners(load_preprocd_data=False)
            rvc.load_preprocd_data(verbose=True)
            out, _ = capfd.readouterr()
            assert "Loading " in out and "preprocd_data.pkl" in out and "Done." in out

        rvc = RoboticVacuumCleaners(load_preprocd_data=False)
        rvc.load_preprocd_data(verified_reviews_only=verified_reviews_only, verbose=True)
        if verified_reviews_only:
            assert len(rvc.preprocd_data) >= 89988
        else:
            assert len(rvc.preprocd_data) >= 101608
        assert rvc.preprocd_data_ is None

        rvc.load_preprocd_data(dual_scale=dual_scale, verbose=True)
        if dual_scale:
            assert len(rvc.preprocd_data) >= 77773
            assert isinstance(rvc.preprocd_data_, pd.DataFrame)
            assert len(rvc.preprocd_data_) >= 101608
        else:
            assert len(rvc.preprocd_data) >= 101608

        del rvc
        gc.collect()


class TestTradVac:

    @pytest.mark.parametrize('use_db', [True, False])
    def test_load_raw_data(self, capfd, use_db):
        tvc = TraditionalVacuumCleaners(load_preprocd_data=False, use_db=use_db)
        tvc.load_raw_data(update=True, verbose=True)
        out, _ = capfd.readouterr()
        assert f"Reading the raw data of {tvc.PRODUCT_NAME.lower()} reviews ... " in out
        if use_db:
            assert ('Importing data into the table "amazon_reviews"."vacuum_cleaners_traditional"'
                    in out)
        else:
            assert '"raw_data.pkl"' in out

        tvc.load_raw_data(verbose=True)
        out, _ = capfd.readouterr()
        assert 'The raw data is already loaded.' in out

        assert isinstance(tvc.raw_data, pd.DataFrame)
        assert len(tvc.raw_data) >= 230479

        del tvc
        gc.collect()

    @pytest.mark.parametrize('use_db', [True, False])
    def test_load_prep_data(self, capfd, use_db):
        tvr = TraditionalVacuumCleaners(load_preprocd_data=False, use_db=use_db)
        tvr.load_prep_data(update=True, verbose=True)
        out, _ = capfd.readouterr()
        assert "Making data ready for preprocessing ... " in out
        if use_db:
            assert ('Importing data into the table '
                    '"amazon_reviews"."vacuum_cleaners_traditional_prep"') in out
        else:
            assert '"prep_data.pkl"' in out

        tvr.load_prep_data(verbose=True)
        out, _ = capfd.readouterr()
        assert 'The preparatory data is already loaded.' in out

        assert isinstance(tvr.prep_data, pd.DataFrame)
        assert len(tvr.prep_data) >= 230478

        del tvr
        gc.collect()

    @pytest.mark.parametrize('use_db', [True, False])
    @pytest.mark.parametrize('verified_reviews_only', [True, False])
    @pytest.mark.parametrize('dual_scale', [True, False])
    def test_load_preprocd_data(self, capfd, use_db, verified_reviews_only, dual_scale):
        tvr = TraditionalVacuumCleaners(load_preprocd_data=False, use_db=use_db)
        tvr.load_preprocd_data(update=True, verbose=True)  # Update preprocd_data
        out, _ = capfd.readouterr()
        assert '\n'.join(preprocd_prt_msg) in out.replace('\x1b[0m', '')

        if use_db:
            m = ('Importing data into the table '
                 '"amazon_reviews"."vacuum_cleaners_traditional_preprocd"')
            assert m in out
        else:
            assert '"preprocd_data.pkl"' in out

        tvr.load_preprocd_data(verbose=True)
        out, _ = capfd.readouterr()
        assert 'The preprocessed data is already loaded.' in out

        tvr.load_preprocd_data(verified_reviews_only=verified_reviews_only)
        if verified_reviews_only:
            assert len(tvr.preprocd_data) >= 131971
        else:
            assert len(tvr.preprocd_data) >= 146628

        tvr.load_preprocd_data(dual_scale=dual_scale)
        if dual_scale:
            assert len(tvr.preprocd_data) >= 110978
            assert isinstance(tvr.preprocd_data_, pd.DataFrame)
            assert len(tvr.preprocd_data_) >= 146656
        else:
            assert len(tvr.preprocd_data) >= 146628


class TestSmartThermostats:

    @pytest.mark.parametrize('use_db', [True, False])
    def test_load_raw_data(self, capfd, use_db):
        smt = SmartThermostats(load_preprocd_data=False, use_db=use_db)
        smt.load_raw_data(update=True, verbose=True)
        out, _ = capfd.readouterr()
        assert f"Reading the raw data of {smt.PRODUCT_NAME.lower()} reviews ... " in out

        if use_db:
            assert 'Importing data into the table "amazon_reviews"."thermostats_smart"' in out

            smt = SmartThermostats(load_preprocd_data=False, use_db=use_db)
            smt.load_raw_data(verbose=True)
            out, _ = capfd.readouterr()
            assert (f"Loading the raw data of the {smt.PRODUCT_NAME.lower()} reviews ... " in out
                    and "Done." in out)

            smt.load_raw_data(verbose=True)
            out, _ = capfd.readouterr()
            assert "The raw data is already loaded." in out

        else:
            assert '"raw_data.pkl"' in out and "Done." in out

            smt = SmartThermostats(load_preprocd_data=False, use_db=use_db)
            smt.load_raw_data(verbose=True)
            out, _ = capfd.readouterr()
            assert "Loading " in out and "raw_data.pkl" in out and "Done." in out

        assert isinstance(smt.raw_data, pd.DataFrame)
        assert len(smt.raw_data) >= 50835

        del smt
        gc.collect()


if __name__ == '__main__':
    pytest.main()
