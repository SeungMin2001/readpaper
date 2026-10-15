"""Browser checks against an isolated fixture server; never calls the real answer LLM."""
import asyncio,json,os,sys,socket,subprocess,tempfile
import httpx
from pathlib import Path
import pymupdf
from playwright.async_api import async_playwright
REPO_ROOT=Path(__file__).resolve().parents[2]
root=REPO_ROOT/'data'/'inspection'/'ui-check';root.mkdir(parents=True,exist_ok=True)
doc=pymupdf.open();page=doc.new_page(width=612,height=792)
doc.set_metadata({'title':'Attention Is All You Need'})
page.insert_text((50,60),'Attention Is All You Need',fontsize=22)
page.insert_text((50,93),'Ashish Vaswani et al. | Research paper',fontsize=11)
page.insert_text((50,140),'Abstract',fontsize=14)
lines=['Multi-head attention allows the model to jointly attend to information',
'from different representation subspaces at different positions.',
'The Transformer uses attention mechanisms without recurrence.',
'Queries and keys determine how much each value contributes.',
'We evaluate our model on machine translation tasks and compare',
'its performance with recurrent and convolutional neural networks.']
for i,line in enumerate(lines):page.insert_text((50,165+i*20),line,fontsize=11)
page.insert_text((50,330),'1  The Transformer architecture',fontsize=15)
for i,word in enumerate(['Input','Multi-head attention','Feed forward','Output']):
    x=65+(i%2)*250;y=370+(i//2)*90
    page.draw_rect(pymupdf.Rect(x,y,x+200,y+50),color=(.45,.5,.55),fill=(.95,.96,.97))
    page.insert_text((x+15,y+30),word,fontsize=12)
page.insert_text((50,580),'Figure 1: Attention-based sequence modeling architecture.',fontsize=10)
page.insert_text((50,630),'The attention function maps a query and a set of key-value pairs',fontsize=11)
page.insert_text((50,650),'to an output. The output is computed as a weighted sum of values.',fontsize=11)
page.insert_text((50,730),'1',fontsize=10)
file=root/'sample.pdf';doc.save(file);doc.close()
async def main(base_url):
 async with async_playwright() as p:
    browser=await p.chromium.launch(headless=True,args=['--no-sandbox'])
    tab=await browser.new_page(viewport={'width':1512,'height':982},device_scale_factor=1)
    errors=[];tab.on('pageerror',lambda e:errors.append(str(e)))
    await tab.goto(base_url)
    await tab.wait_for_selector('.upload-card')
    await tab.screenshot(path=str(root/'welcome.png'),full_page=True)
    await tab.locator('input[type=file]').set_input_files(str(file))
    await tab.wait_for_selector('.workspace',timeout=30000)
    await tab.wait_for_selector('.pdf-word')
    await tab.screenshot(path=str(root/'reader.png'),full_page=True)
    word=tab.get_by_role('button',name='Multi-head 뜻 보기',exact=True).first
    await word.click();await tab.wait_for_selector('.pdf-gloss')
    bounds=await tab.locator('.pdf-gloss').evaluate_all('(nodes)=>nodes.map(n=>({width:n.clientWidth,scroll:n.scrollWidth,height:n.clientHeight,text:n.textContent}))')
    assert all(b['width']>=b['scroll'] for b in bounds),bounds
    await tab.screenshot(path=str(root/'translation.png'),full_page=True)
    await tab.get_by_role('button',name='문장 선택',exact=True).click()
    first=await tab.get_by_role('button',name='Multi-head 뜻 보기',exact=True).first.bounding_box()
    last=await tab.get_by_role('button',name='information 뜻 보기',exact=True).bounding_box()
    await tab.mouse.move(first['x']+2,first['y']+first['height']/2);await tab.mouse.down()
    await tab.mouse.move(last['x']+last['width']-2,last['y']+last['height']/2,steps=12);await tab.mouse.up()
    await tab.wait_for_selector('.selection-preview')
    await tab.locator('.selection-preview').drag_to(tab.locator('.assistant-panel'))
    await tab.wait_for_selector('.attachment-chip')
    assert 'Multi-head attention' in await tab.locator('.attachment-chip').inner_text()
    await tab.get_by_role('textbox',name='질문',exact=True).fill('선택한 문장을 설명해줘')
    await tab.get_by_role('button',name='질문 보내기',exact=True).click()
    await tab.wait_for_selector('.answer-sources')
    await tab.locator('.assistant-bubble a[href^="#evidence-"]').first.click()
    assert await tab.locator('.answer-sources').first.get_attribute('open') is not None
    assert '1쪽' in await tab.locator('.answer-sources').inner_text()
    assert 'Multi-head attention' in await tab.locator('.source-excerpt').inner_text()
    await tab.screenshot(path=str(root/'chat.png'),full_page=True)
    await tab.get_by_role('button',name='그림 선택',exact=True).click()
    pdf=await tab.locator('.pdf-page').bounding_box();scale=pdf['width']/612
    await tab.mouse.move(pdf['x']+60*scale,pdf['y']+355*scale);await tab.mouse.down()
    await tab.mouse.move(pdf['x']+520*scale,pdf['y']+530*scale,steps=10);await tab.mouse.up()
    await tab.wait_for_selector('.selection-preview img')
    await tab.get_by_role('button',name='채팅에 첨부 ↗').click()
    await tab.get_by_role('textbox',name='질문',exact=True).fill('이 그림을 설명해줘')
    await tab.get_by_role('button',name='질문 보내기',exact=True).click()
    await tab.wait_for_function("document.querySelectorAll('.answer-sources').length===2")
    await tab.screenshot(path=str(root/'figure-chat.png'),full_page=True)
    await tab.get_by_role('tab',name='인용 지도').click()
    await tab.wait_for_selector('.citation-graph')
    assert await tab.locator('.graph-node').count()==8
    await tab.get_by_role('button',name='크게 보기 ↗').click()
    await tab.screenshot(path=str(root/'graph.png'),full_page=True)
    await tab.get_by_role('button',name='닫기 ×').click()
    await tab.get_by_role('tab',name='대화',exact=True).click()
    await tab.set_viewport_size({'width':390,'height':844})
    await tab.screenshot(path=str(root/'mobile.png'),full_page=True)
    overflow=await tab.evaluate('document.documentElement.scrollWidth > window.innerWidth')
    assert not overflow,'mobile horizontal overflow'
    assert not errors,errors
    print(json.dumps({'browser_errors':errors,'translation_bounds':bounds,'checks':['upload','word-centering-bounds','text-drag','attachment','chat-source-page-and-excerpt','figure-crop-and-chat','graph','mobile-no-overflow']},ensure_ascii=False))
    await browser.close()
async def check():
    with tempfile.TemporaryDirectory(prefix='readpaper-ui-') as directory:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            port=sock.getsockname()[1]
        env=os.environ | {'READPAPER_TEST_DATA_DIR': directory, 'READPAPER_TEST_PORT': str(port)}
        with (root/'server.log').open('w') as output:
            process=subprocess.Popen([sys.executable,str(REPO_ROOT/'backend'/'tests'/'workspace_fixture_server.py')],env=env,stdout=output,stderr=subprocess.STDOUT)
            try:
                base_url=f'http://127.0.0.1:{port}'
                async with httpx.AsyncClient() as client:
                    for _ in range(100):
                        if process.poll() is not None:
                            raise RuntimeError('Fixture server failed; see data/inspection/ui-check/server.log')
                        try:
                            if (await client.get(base_url+'/api/health')).status_code==200:
                                break
                        except httpx.HTTPError:
                            pass
                        await asyncio.sleep(.2)
                    else:
                        raise RuntimeError('Fixture server did not become ready')
                await main(base_url)
            finally:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()

if __name__=='__main__': asyncio.run(check())
