from pathlib import Path
import pytest
import matplotlib.pyplot as plt
import gwak.sky_position_antenna_patterns as sky

ARGS=(138.47,-1.94,'2011-05-14T01:32:22.654109')


def test_temp_modes_show_without_saving_and_preserve_results(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    shown=[]
    def display(fig):
        fig.canvas.draw()
        shown.append(len(fig.axes))
        plt.close(fig)
    monkeypatch.setattr(sky,'_show_temporary_figure',display)
    def forbid(*args,**kwargs):raise AssertionError('Temporary plot tried to write a file')
    monkeypatch.setattr('matplotlib.figure.Figure.savefig',forbid)
    monkeypatch.setattr(Path,'write_text',forbid)
    a=sky.event_to_gwak(*ARGS)
    assert sky.event_to_gwak(*ARGS,temp_plot=True)==a
    direct=sky.event_to_gwak_from_first_principles(*ARGS)
    assert sky.event_to_gwak_from_first_principles(*ARGS,temp_plot=True)==direct
    response=sky.antenna_patterns(a)
    assert sky.antenna_patterns(a,temp_plot=True)==response
    assert sky.plot_antenna_patterns(a,output=tmp_path/'unused'/'x',temp_plot=True)==response
    assert sky.plot_event_position(*ARGS,output=tmp_path/'unused'/'x',temp_plot=True)==dict(a,name='Event')
    assert shown==[2,2,12,12,2]
    assert list(tmp_path.iterdir())==[]
    assert not plt.get_fignums()


def test_default_calculations_do_not_plot(monkeypatch):
    def forbid(*a,**kw):raise AssertionError('Unexpected plot')
    monkeypatch.setattr(sky,'_position_figure',forbid)
    monkeypatch.setattr(sky,'_antenna_figure',forbid)
    sky.antenna_patterns(sky.event_to_gwak(*ARGS))
    sky.antenna_patterns(sky.event_to_gwak_from_first_principles(*ARGS))


def test_headless_error_cleans_up():
    fig=plt.figure()
    with pytest.raises(RuntimeError,match='interactive Matplotlib backend'):
        sky._show_temporary_figure(fig)
    assert not plt.fignum_exists(fig.number)


def test_gui_display_blocks_and_cleans_up(monkeypatch):
    fig=plt.figure()
    monkeypatch.setattr(fig.canvas,'required_interactive_framework','test')
    calls=[]
    monkeypatch.setattr(plt,'show',lambda **kw:calls.append(kw))
    sky._show_temporary_figure(fig)
    assert calls==[{'block':True}]
    assert not plt.fignum_exists(fig.number)


def test_cli_temp_does_not_write(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('sys.argv',['sky','--ra-deg','138.47','--dec-deg','-1.94',
        '--trigger-time',ARGS[2],'--temp-plot','--antenna-patterns','--output','unused/path'])
    shown=[]
    def display(fig):shown.append(fig);plt.close(fig)
    monkeypatch.setattr(sky,'_show_temporary_figure',display)
    sky.main()
    assert len(shown)==2
    assert list(tmp_path.iterdir())==[]
